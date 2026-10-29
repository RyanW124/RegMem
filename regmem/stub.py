"""Deterministic stand-in for a language model, for dry runs with no GPU or model weights.

    python -m regmem consolidate --model stub ...

Every response is a pure function of the prompt, so a pipeline run is reproducible byte for byte.
The double answers every schema this package defines; anything else gets one generic filler field
per required pydantic field. `StubLLM` is deliberately not smart: it exercises the plumbing
(proposal validation, repair loop, traces, batching), not the quality of the memory it builds.
"""
import hashlib
import json
from typing import TypeVar

from pydantic import BaseModel

from .typedefs import (ConsolidationOutput, JudgeResult, MemoryOperation, RegressionSuiteOperation,
                      RegressionSuiteOutput)

PROBES_PER_CALL = 4
T = TypeVar('T', bound=BaseModel)


class StubLLM:
    """Answer/propose/judge double whose output depends only on the prompt text."""

    @staticmethod
    def _roll(text: str) -> int:
        """Stable pseudo-random number in [0, 100) for a prompt."""
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % 100

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_batch(system_prompt, [user_prompt])[0]

    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T:
        return self.generate_structured_batch(system_prompt, [user_prompt], schema)[0]

    def generate_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str]:
        return [f'stub answer {self._roll(prompt)}' for prompt in user_prompts]

    def generate_structured_batch(self, system_prompt: str, user_prompts: list[str], schema: type[T]) -> list[T]:
        return [self._fake(prompt, schema) for prompt in user_prompts]

    def _fake(self, prompt: str, schema: type[T]) -> T:
        roll = self._roll(prompt)
        if schema is RegressionSuiteOutput:
            return schema(operations=[
                RegressionSuiteOperation(action='WRITE', target_id=None, query=f'stub probe {roll}.{n}?',
                                         expected_answer=f'stub {roll}.{n}')
                for n in range(PROBES_PER_CALL)])
        if schema is ConsolidationOutput:
            payload = json.loads(prompt)
            # The repair payload must carry the answer memory gave, never the reference answer or
            # the judge's reason: those would let a repair copy gold and pass without the fact.
            if any({'expected_answer', 'reason'} & set(failure) for failure in payload.get('failed_tests', [])):
                raise SystemExit('gold answer leaked into the repair payload')  # not caught by the runner
            operations = [MemoryOperation(action='WRITE', target_id=None, content=f'stub fact {roll}')]
            if payload['memory'] and roll < 30:
                operations.append(MemoryOperation(action='PRUNE', target_id=payload['memory'][0]['id'], content=None))
            if roll >= 97:  # an invalid op now and then, to exercise error handling
                operations.append(MemoryOperation(action='REVISE', target_id='no-such-id', content='x'))
            return schema(operations=operations)
        if schema is JudgeResult:
            return schema(passed=roll < 85, reason='stub verdict')
        # Judge schemas defined by callers (e.g. the MemConflict benchmark) name their own metrics.
        return schema(**{name: 'stub' if name == 'reasoning'
                         else (1.0 if roll < 50 else 0.0) if 'accuracy' in name else int(roll < 50)
                         for name in schema.model_fields})
