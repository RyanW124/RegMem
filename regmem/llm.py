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
    ):
        if temperature < 0:
            raise ValueError("Temperature must be nonnegative")

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

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

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
        )

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

        if schema not in self._generators:
            self._generators[schema] = Generator(
                self.outlines_model,
                schema,
            )

        response = self._generators[schema](
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        print("u: ", response)
        return schema.model_validate_json(response)
