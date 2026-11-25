"""Transactional propose → verify → bounded repair → accept/reject."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable
from uuid import uuid4
from .proposer import Consolidator, apply_memory_operations
from .regression import RegressionGenerator, SuiteCurator, apply_regression_operations
from .typedefs import ConsolidationResult, MemoryState, RegressionSuite, VerificationResult
from .verifier import TransitionChecker, Verifier

class ConsolidationEngine:
    """With no tests and no transition checker this is unverified consolidation; the token budget,
    when set, is checked the same way for every variant."""
    def __init__(self, consolidator: Consolidator, regression_generator: RegressionGenerator,
                 verifier: Verifier, max_repairs: int = 2, *, curator: SuiteCurator | None = None,
                 transition_checker: TransitionChecker | None = None, token_budget: int | None = None,
                 count_tokens: Callable[[MemoryState], int] | None = None, test_budget: int | None = None):
        if max_repairs < 0:
            raise ValueError('max_repairs must be nonnegative')
        if token_budget is not None and count_tokens is None:
            raise ValueError('token_budget needs count_tokens')
        self.consolidator = consolidator
        self.regression_generator = regression_generator
        self.verifier = verifier
        self.max_repairs = max_repairs
        self.curator = curator
        self.transition_checker = transition_checker
        self.token_budget = token_budget
        self.count_tokens = count_tokens
        self.test_budget = test_budget

    def consolidate(self, memory: MemoryState, suite: RegressionSuite, evidence: str,
                    *, save_run: bool = False,
                    output_dir: str | Path = 'results/runs') -> ConsolidationResult:
        """Optionally save a JSON trace, including rejected and errored runs.

        Saving is disabled by default. File-system errors propagate to the caller.
        """
        trace = None
        if save_run:
            trace = {'schema_version': 1,
                     'started_at': datetime.now(timezone.utc).isoformat(),
                     'max_repairs': self.max_repairs, 'evidence': evidence,
                     'initial_memory': asdict(memory), 'initial_suite': asdict(suite),
                     'events': []}
        try:
            result = self._consolidate(memory, suite, evidence, trace)
        except Exception as error:
            if trace is not None:
                trace.update(status='error', error={'type': type(error).__name__, 'message': str(error)})
                self._save_trace(trace, output_dir)
            raise
        if trace is not None:
            trace.update(status='accepted' if result.accepted else 'rejected',
                         final_memory=asdict(result.memory), final_suite=asdict(result.suite),
                         attempts=result.attempts)
            result.run_path = self._save_trace(trace, output_dir)
        return result

    @staticmethod
    def _save_trace(trace: dict, output_dir: str | Path) -> str:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        trace['finished_at'] = datetime.now(timezone.utc).isoformat()
        path = directory / f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}-{uuid4().hex}.json"
        # Exclusive creation and a random filename prevent concurrent runs overwriting each other.
        with path.open('x', encoding='utf-8') as stream:
            json.dump(trace, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return str(path.resolve())

    @staticmethod
    def _attempt(propose, record, attempt):
        # A malformed proposal (invalid JSON or an op on an unknown id) is an empty patch, so
        # verification fails and the bounded repair loop retries instead of aborting the cycle.
        # pydantic's ValidationError subclasses ValueError.
        try:
            return list(propose())
        except ValueError as error:
            record('invalid_proposal', attempt=attempt, error=f'{type(error).__name__}: {error}')
            return []

    def _budget(self, memory, over_by=None):
        if self.token_budget is None:
            return None
        budget = {'memory_budget_tokens': self.token_budget, 'memory_tokens': self.count_tokens(memory)}
        return budget if over_by is None else {**budget, 'over_budget_by_tokens': over_by}

    def _consolidate(self, memory, suite, evidence, trace):
        def record(event, **details):
            if trace is not None:
                trace['events'].append({'event': event, **details})

        # Freeze the expanded suite before the memory proposal, so neither proposal nor repair can weaken it.
        test_operations = self.regression_generator.propose(suite, evidence)
        record('regression_proposal', operations=[op.model_dump() for op in test_operations])
        expanded = apply_regression_operations(suite, test_operations)
        next_id = expanded.next_id
        if self.curator is not None and expanded.tests:
            edits = self._attempt(lambda: self.curator.curate(expanded, evidence, suite.next_id, self.test_budget),
                                  record, attempt=0)
            record('suite_edits', operations=[op.model_dump() for op in edits])
            expanded = apply_regression_operations(expanded, edits)
        if self.test_budget is not None and len(expanded.tests) > self.test_budget:
            # The curator left the suite over budget: drop the newest tests, which protect nothing yet.
            expanded.tests = expanded.tests[:self.test_budget]
        # A regression is a test the current memory passes. The suite keeps only tests the accepted
        # memory passed, so exactly the unchanged old tests are protected; this session's new tests and
        # revised tests do not block, and join the suite only if the accepted memory passes them.
        before = {t.id: t.expected_answer for t in suite.tests}
        protected = {t.id for t in expanded.tests if before.get(t.id) == t.expected_answer}
        record('expanded_suite', suite=asdict(expanded), protected=sorted(protected, key=int))
        operations = self._attempt(lambda: self.consolidator.propose(memory, evidence, self._budget(memory)),
                                   record, attempt=1)
        record('memory_proposal', operations=[op.model_dump() for op in operations])
        candidate = apply_memory_operations(memory, operations)
        record('candidate', attempt=1, memory=asdict(candidate))
        by_id = {t.id: t for t in expanded.tests}
        for attempt in range(self.max_repairs + 1):
            verification = (self.verifier.verify(candidate, expanded) if expanded.tests else VerificationResult([]))
            broken = [r for r in verification.results if not r.passed and r.test_id in protected]
            problems = (self.transition_checker.check(memory, evidence, candidate, self.token_budget)
                        if self.transition_checker is not None else [])
            over_by = max(0, self.count_tokens(candidate) - self.token_budget) if self.token_budget is not None else 0
            blocking = {'broken_test_ids': [r.test_id for r in broken], 'problems': problems, 'over_budget_by': over_by}
            record('verification', attempt=attempt + 1, passed=not (broken or problems or over_by),
                   failed_test_ids=verification.failed_test_ids, **blocking,
                   results=[asdict(r) for r in verification.results])
            if not (broken or problems or over_by):
                # All internal patches form one externally visible transaction.
                candidate.version = memory.version + int(bool(operations))
                kept = deepcopy(expanded)
                kept.tests = [t for t in kept.tests if t.id not in set(verification.failed_test_ids)]
                return ConsolidationResult(True, candidate, kept, verification, attempt + 1, operations,
                                           blocking=blocking)
            if attempt < self.max_repairs:
                # Repair sees what memory answered, never the reference answer or the judge's
                # reason (written with the reference in view), so it cannot copy gold to pass.
                failures = ([{'test_id': r.test_id, 'query': by_id[r.test_id].query,
                              'predicted_answer': r.predicted_answer} for r in broken]
                            + [{'problem': problem} for problem in problems])
                # The last accepted memory passed the protected tests, so regressions usually trace to
                # items this cycle revised or pruned: hand repair their pre-cycle versions.
                current = {x.id: x.content for x in candidate.items}
                previous_items = [asdict(x) for x in memory.items if current.get(x.id) != x.content]
                patch = self._attempt(lambda: self.consolidator.repair(candidate, evidence, failures, previous_items,
                                                                       self._budget(candidate, over_by)),
                                      record, attempt=attempt + 2)
                record('repair_patch', after_attempt=attempt + 1, failures=failures,
                       previous_items=previous_items, operations=[op.model_dump() for op in patch])
                candidate = apply_memory_operations(candidate, patch)
                record('candidate', attempt=attempt + 2, memory=asdict(candidate))
                operations.extend(patch)
        # Reject rolls back memory and suite. next_id still advances, so test ids keep matching the
        # order the probes were generated in.
        rolled_back = deepcopy(suite)
        rolled_back.next_id = next_id
        return ConsolidationResult(False, deepcopy(memory), rolled_back, verification,
                                   self.max_repairs + 1, operations, blocking=blocking)
