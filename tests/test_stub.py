"""The deterministic double: stable output, schema-shaped responses, and a full pipeline run."""
import json

import pytest

from helpers import memory, suite

from regmem import ConsolidationEngine, Consolidator, RegressionGenerator, StubLLM, Verifier
from regmem.stub import PROBES_PER_CALL
from regmem.typedefs import ConsolidationOutput, JudgeResult, MemoryState, RegressionSuite, RegressionSuiteOutput


def test_output_is_deterministic():
    first, second = StubLLM(), StubLLM()
    assert first.generate_batch('sys', ['a', 'b']) == second.generate_batch('sys', ['a', 'b'])
    assert first.generate('sys', 'a') != first.generate('sys', 'b')


def test_a_batch_answers_every_prompt():
    stub = StubLLM()
    prompts = [f'prompt {n}' for n in range(7)]
    assert len(stub.generate_batch('sys', prompts)) == len(prompts)


def test_probe_proposals_are_writes_with_a_query_and_an_answer():
    operations = StubLLM().generate_structured('sys', 'evidence', RegressionSuiteOutput).operations
    assert len(operations) == PROBES_PER_CALL
    for operation in operations:
        assert operation.action == 'WRITE' and operation.target_id is None
        assert operation.query and operation.expected_answer


def test_memory_proposals_always_write_and_sometimes_prune():
    proposals = [StubLLM().generate_structured('sys', json.dumps({'memory': [{'id': '0', 'content': 'x'}],
                                                                  'new_evidence': f'e{n}'}), ConsolidationOutput)
                 for n in range(40)]
    actions = {operation.action for proposal in proposals for operation in proposal.operations}
    # WRITE always, PRUNE sometimes, and a REVISE on an unknown id now and then so the engine's
    # invalid-proposal path is exercised.
    assert actions == {'WRITE', 'PRUNE', 'REVISE'}
    assert all(proposal.operations[0].action == 'WRITE' for proposal in proposals)


def test_the_verdict_is_a_boolean():
    verdict = StubLLM().generate_structured('sys', '{"query": "q", "expected_answer": "a", '
                                                        '"predicted_answer": "a"}', JudgeResult)
    assert isinstance(verdict, JudgeResult) and isinstance(verdict.passed, bool)


def test_the_stub_refuses_a_repair_payload_carrying_gold():
    payload = json.dumps({'memory': [], 'new_evidence': 'e',
                          'failed_tests': [{'test_id': '0', 'query': 'q', 'expected_answer': 'a'}]})
    with pytest.raises(SystemExit, match='gold answer leaked'):
        StubLLM().generate_structured('sys', payload, ConsolidationOutput)


def test_a_full_cycle_runs_and_reports_a_verdict(tmp_path):
    stub = StubLLM()
    engine = ConsolidationEngine(Consolidator(stub), RegressionGenerator(stub), Verifier(stub), max_repairs=1)
    result = engine.consolidate(memory('stub fact 0'), suite(('a?', 'b')), 'I prefer tea.',
                                save_run=True, output_dir=tmp_path)
    assert isinstance(result.accepted, bool)
    assert result.attempts >= 1 and result.verification.results
    assert result.run_path and result.memory.version >= 0


def test_the_stub_satisfies_the_protocol_for_empty_input():
    stub = StubLLM()
    assert stub.generate_batch('sys', []) == []
    assert stub.generate_structured_batch('sys', [], JudgeResult) == []
    assert stub.generate_batch('sys', ['x'])[0].startswith('stub answer ')
    assert isinstance(RegressionGenerator(stub).propose(RegressionSuite(), 'e')[0].action, str)
    assert Verifier(stub).verify(MemoryState(), RegressionSuite()).passed
