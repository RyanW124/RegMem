"""Wrapper for LLMs."""


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
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

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

        return self.outlines_model(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ).strip()

    @torch.inference_mode()
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
    ) -> T:
        prompt = self._format_prompt(system_prompt, user_prompt)

        response = self._generator(schema)(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return schema.model_validate_json(response)

    @torch.inference_mode()
    def generate_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
    ) -> list[str]:
        prompts = [self._format_prompt(system_prompt, p) for p in user_prompts]

        outputs = self._run_batched(lambda chunk: self.outlines_model.batch(
            chunk,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ), prompts)
        return [output.strip() for output in outputs]

    @torch.inference_mode()
    def generate_structured_batch(
        self,
        system_prompt: str,
        user_prompts: list[str],
        schema: type[T],
    ) -> list[T]:
        prompts = [self._format_prompt(system_prompt, p) for p in user_prompts]
        generator = self._generator(schema)

        responses = self._run_batched(lambda chunk: generator.batch(
            chunk,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ), prompts)
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
