"""Memory, cumulative regression tests, and transaction results."""
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar
from pydantic import BaseModel, ConfigDict, Field

Action = Literal['WRITE', 'REVISE', 'PRUNE']
T = TypeVar('T', bound=BaseModel)

class LanguageModel(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...
    def generate_structured(self, system_prompt: str, user_prompt: str, schema: type[T]) -> T: ...

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

class MemoryOperation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Action
    target_id: str | None = None
    content: str | None = None


class ConsolidationOutput(BaseModel):
    operations: list[MemoryOperation] = Field(default_factory=list)

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
    target_id: str | None = None
    query: str | None = None
    expected_answer: str | None = None

class RegressionSuiteOutput(BaseModel):
    operations: list[RegressionSuiteOperation] = Field(default_factory=list)

class JudgeResult(BaseModel):
    passed: bool
    reason: str

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
