"""Memory, cumulative regression tests, and transaction results."""
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar
from pydantic import BaseModel, ConfigDict, Field

Action = Literal['WRITE', 'REVISE', 'PRUNE']
T = TypeVar('T', bound=BaseModel)

class LanguageModel(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T: ...
    def generate_batch(self, system_prompt: str, user_prompts: list[str]) -> list[str]: ...
    def generate_structured_batch(self, system_prompt: str, user_prompts: list[str], schema: type[T]) -> list[T]: ...

@dataclass
class MemoryItem:
    id: str
    content: str
    source_ids: list[str] = field(default_factory=list) #ignore for now

@dataclass
class MemoryState:
    items: list[MemoryItem] = field(default_factory=list)
    version: int = 0
    next_id: int = 0

    def __len__(self):
        return len(self.items)

# Output schemas keep every field required (null where allowed): under constrained decoding an
# optional field lets the model emit `{}` or skip content entirely. Length caps keep outputs
# inside max_new_tokens and memory items atomic.
class MemoryOperation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Action
    target_id: str | None
    content: str | None = Field(max_length=300)


class ConsolidationOutput(BaseModel):
    operations: list[MemoryOperation]

@dataclass
class DecisionCase:
    id: str
    query: str
    expected_answer: str

@dataclass
class RegressionSuite:
    tests: list[DecisionCase] = field(default_factory=list)
    version: int = 0
    next_id: int = 0

class RegressionSuiteOperation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Action
    target_id: str | None
    query: str | None = Field(max_length=200)
    expected_answer: str | None = Field(max_length=150)

class RegressionSuiteOutput(BaseModel):
    operations: list[RegressionSuiteOperation]

class JudgeResult(BaseModel):
    # Fields generate in order under constrained decoding: rationale first, then the verdict.
    reason: str = Field(max_length=300)
    passed: bool

@dataclass
class TestResult:
    test_id: str
    predicted_answer: str
    expected_answer: str
    passed: bool
    reason: str = ''

@dataclass
class VerificationResult:
    results: list[TestResult]

    @property
    def passed(self):
        return all(result.passed for result in self.results)

    @property
    def failed_test_ids(self):
        return [result.test_id for result in self.results if not result.passed]

@dataclass
class ConsolidationResult:
    accepted: bool
    memory: MemoryState
    suite: RegressionSuite
    verification: VerificationResult
    attempts: int
    operations: list[MemoryOperation] = field(default_factory=list)
    run_path: str | None = None
