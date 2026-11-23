"""Wrapper for LLMs."""


from contextlib import contextmanager
from copy import copy
from threading import RLock
from typing import TypeVar

import torch
import outlines
from outlines import Generator
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


T = TypeVar("T", bound=BaseModel)


class LLM:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.01,
        batch_size: int = 8,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        model.eval()
        self._setup(model, tokenizer, max_new_tokens, temperature, batch_size)

    @classmethod
    def from_model(cls, model, tokenizer, *, max_new_tokens=256, temperature=0.0, batch_size=2):
        """Wrap an already loaded model without changing its training mode or weights."""
        llm = cls.__new__(cls)
        llm._setup(model, tokenizer, max_new_tokens, temperature, batch_size)
        return llm

    def _setup(self, model, tokenizer, max_new_tokens, temperature, batch_size):
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        if batch_size < 1 or max_new_tokens < 1:
            raise ValueError("Batch size and max_new_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer

        self.outlines_model = outlines.from_transformers(
            self.model,
            self.tokenizer,
        )
        # Prompts are already chat-formatted by _format_prompt; stop Outlines
        # (1.3.3) from wrapping them in the chat template a second time.
        self.outlines_model.type_adapter.has_chat_template = False

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        # Reuse generators for repeated calls with the same schema.
        self._generators = {}
        self._lock = RLock()
        self._adapter_name = None
        self._adapter_paths = {}

    def for_adapter(self, path: str, name: str):
        """Return an inference view sharing the base weights, tokenizer, and generator cache."""
        with self._lock:
            if name in self._adapter_paths:
                if self._adapter_paths[name] != str(path):
                    raise ValueError(f'Adapter {name} is already loaded from {self._adapter_paths[name]}')
            else:
                try:
                    self.model.load_adapter(str(path), adapter_name=name, is_trainable=False)
                finally:
                    if getattr(self.model, '_hf_peft_config_loaded', False):
                        self.model.disable_adapters()
                self._adapter_paths[name] = str(path)
            view = copy(self)
            view._adapter_name = name
        return view

    @contextmanager
    def _generation_context(self):
        # Adapter selection mutates the shared model. Serialize base calls with role calls.
        with self._lock:
            if self._adapter_name is None:
                yield
                return
            try:
                self.model.set_adapter(self._adapter_name)
                self.model.enable_adapters()
                yield
            finally:
                self.model.disable_adapters()

    def _generation_kwargs(self):
        if self.temperature == 0:
            return {'max_new_tokens': self.max_new_tokens, 'do_sample': False}
        return {'max_new_tokens': self.max_new_tokens, 'temperature': self.temperature}

    def _format_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen3 thinks by default; other chat templates ignore this flag.
            enable_thinking=False,
        )

    def _generator(self, schema: type[T]):
        if schema not in self._generators:
            self._generators[schema] = Generator(
                self.outlines_model,
                schema,
            )
        return self._generators[schema]

    @torch.inference_mode()
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        prompt = self._format_prompt(system_prompt, user_prompt)

        with self._generation_context():
            return self.outlines_model(prompt, **self._generation_kwargs()).strip()

    @torch.inference_mode()
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        prompt = self._format_prompt(system_prompt, user_prompt)

        with self._generation_context():
            response = self._generator(schema)(prompt, **self._generation_kwargs())
        return schema.model_validate_json(response)

    @torch.inference_mode()
    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
    ) -> list[str]:
        prompts = [self._format_prompt(system_prompt, p) for p in user_prompts]

        with self._generation_context():
            outputs = self._run_batched(lambda chunk: self.outlines_model.batch(
                chunk, **self._generation_kwargs()), prompts)
        return [output.strip() for output in outputs]

    @torch.inference_mode()
    def generate_structured_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        schema: type[T],
    ) -> list[T]:
        prompts = [self._format_prompt(system_prompt, p) for p in user_prompts]
        with self._generation_context():
            generator = self._generator(schema)
            responses = self._run_batched(lambda chunk: generator.batch(
                chunk, **self._generation_kwargs()), prompts)
        return [schema.model_validate_json(r) for r in responses]

    def _run_batched(self, run, prompts: list[str]) -> list[str]:
        # Late sessions put thousands of memory tokens in every prompt, so a full batch can
        # exceed GPU memory. On out-of-memory, halve the chunk size and retry the same chunk.
        outputs, start, size = [], 0, self.batch_size
        while start < len(prompts):
            chunk = prompts[start:start + size]
            try:
                outputs.extend(run(chunk))
            except torch.cuda.OutOfMemoryError:
                if size == 1:
                    raise
                torch.cuda.empty_cache()
                size //= 2
                continue
            start += len(chunk)
        return outputs
