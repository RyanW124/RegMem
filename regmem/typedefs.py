"""Memory, cumulative regression tests, and transaction results."""
from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol, TypeVar
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
    # Content generates first so the model states the fact before choosing an action and target;
    # action-first made the 8B model REVISE ids 0, 1, 2... in order regardless of content.
    model_config = ConfigDict(extra='forbid')
    content: str | None = Field(max_length=300)
    action: Action
    target_id: str | None


class ConsolidationOutput(BaseModel):
    # The 8B model ignores a prompt-only limit and lists ops until max_new_tokens truncates the JSON.
    # 16 leaves room to add a session's facts and prune enough to stay under a memory budget in one call.
    operations: list[MemoryOperation] = Field(max_length=16)

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

class SuiteEdit(BaseModel):
    # The reason generates first so the model says why before it picks a test and an action.
    model_config = ConfigDict(extra='forbid')
    reason: str = Field(max_length=150)
    target_id: str
    action: Literal['REVISE', 'PRUNE']
    expected_answer: str | None = Field(max_length=150)

class SuiteEditOutput(BaseModel):
    edits: list[SuiteEdit] = Field(max_length=12)

class TransitionVerdict(BaseModel):
    # An empty list passes the update; each problem becomes repair feedback.
    model_config = ConfigDict(extra='forbid')
    problems: list[Annotated[str, Field(max_length=200)]] = Field(max_length=8)

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
    # What blocked the last attempt: broken protected tests, reviewer problems, tokens over budget.
    blocking: dict = field(default_factory=dict)
