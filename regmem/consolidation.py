"""Transactional propose → verify → bounded repair → accept/reject."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
from .proposer import Consolidator, apply_memory_operations
from .regression import RegressionGenerator, apply_regression_operations
from .typedefs import ConsolidationResult, MemoryState, RegressionSuite
from .verifier import Verifier

class ConsolidationEngine:
    def __init__(self, consolidator: Consolidator, regression_generator: RegressionGenerator,
                 verifier: Verifier, max_repairs: int = 2):
        if max_repairs < 0:
            raise ValueError('max_repairs must be nonnegative')
        self.consolidator = consolidator
        self.regression_generator = regression_generator
        self.verifier = verifier
        self.max_repairs = max_repairs

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

    def _consolidate(self, memory, suite, evidence, trace):
        def record(event, **details):
            if trace is not None:
                trace['events'].append({'event': event, **details})

        # Freeze the expanded suite before any repair, so repairs cannot weaken it.
        test_operations = self.regression_generator.propose(suite, evidence)
        record('regression_proposal', operations=[op.model_dump() for op in test_operations])
        expanded = apply_regression_operations(suite, test_operations)
        record('expanded_suite', suite=asdict(expanded))
        operations = list(self.consolidator.propose(memory, evidence))
        record('memory_proposal', operations=[op.model_dump() for op in operations])
        candidate = apply_memory_operations(memory, operations)
        record('candidate', attempt=1, memory=asdict(candidate))
        for attempt in range(self.max_repairs + 1):
            verification = self.verifier.verify(candidate, expanded)
            record('verification', attempt=attempt + 1, passed=verification.passed,
                   failed_test_ids=verification.failed_test_ids,
                   results=[asdict(r) for r in verification.results])
            if verification.passed:
                # All internal patches form one externally visible transaction.
                candidate.version = memory.version + int(bool(operations))
                return ConsolidationResult(True, candidate, expanded, verification, attempt + 1, operations)
            if attempt < self.max_repairs:
                by_id = {t.id: t for t in expanded.tests}
                # Repair sees what memory answered, never the reference answer or the judge's
                # reason (written with the reference in view), so it cannot copy gold to pass.
                failures = [{'test_id': r.test_id, 'query': by_id[r.test_id].query,
                             'predicted_answer': r.predicted_answer}
                            for r in verification.results if not r.passed]
                # The last accepted memory passed the suite, so regressions usually trace to
                # items this cycle revised or pruned: hand repair their pre-cycle versions.
                current = {x.id: x.content for x in candidate.items}
                previous_items = [asdict(x) for x in memory.items if current.get(x.id) != x.content]
                patch = self.consolidator.repair(candidate, evidence, failures, previous_items)
                record('repair_patch', after_attempt=attempt + 1, failures=failures,
                       previous_items=previous_items, operations=[op.model_dump() for op in patch])
                candidate = apply_memory_operations(candidate, patch)
                record('candidate', attempt=attempt + 2, memory=asdict(candidate))
                operations.extend(patch)
        return ConsolidationResult(False, deepcopy(memory), deepcopy(suite), verification,
                                   self.max_repairs + 1, operations)
