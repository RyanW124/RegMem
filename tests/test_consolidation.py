"""The engine is transactional, bounded, and never reports an unchecked candidate as accepted."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from helpers import ScriptedLLM, answers_from, memory, probe, prune, revise, suite, write

from regmem.consolidation import ConsolidationEngine
from regmem.proposer import Consolidator
from regmem.regression import RegressionGenerator
from regmem.typedefs import BatchLengthError
from regmem.verifier import Verifier


def engine_for(llm, max_repairs=2):
    return ConsolidationEngine(Consolidator(llm), RegressionGenerator(llm), Verifier(llm), max_repairs=max_repairs)


def state():
    """One known fact, one probe it passes and one it fails until a WRITE lands."""
    return memory('User was born in Pittsburgh.'), suite(('Where was the user born?', 'Pittsburgh'),
                                                         ('Which drink does the user prefer?', 'tea'))


KEYS = {'Where was the user born?': ('pittsburgh', 'Pittsburgh'),
        'Which drink does the user prefer?': ('tea', 'tea')}


def answer_from_memory(query, items):
    """Answer a probe with its fact when memory holds it, 'unknown' otherwise."""
    needle, answer = KEYS[query]
    contents = ' '.join(item['content'] for item in items).lower()
    return answer if needle in contents else 'unknown'


def accepted_llm(**overrides):
    """Proposes a probe and a WRITE that fixes the failing test, and answers from memory."""
    options = dict(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]],
                   memory_proposals=[[write('User prefers tea.')]], answer=answer_from_memory)
    options.update(overrides)
    return ScriptedLLM(**options)


def test_accepts_a_candidate_that_passes_every_probe():
    memory_before, suite_before = state()
    result = engine_for(accepted_llm()).consolidate(memory_before, suite_before, 'I prefer tea.')
    assert result.accepted and result.attempts == 1
    assert [item.content for item in result.memory.items] == ['User was born in Pittsburgh.', 'User prefers tea.']
    assert result.memory.version == 1 and result.run_path is None
    assert result.verification.passed and result.verification.pass_rate == 1.0
    assert [test.query for test in result.suite.tests] == ['Where was the user born?',
                                                           'Which drink does the user prefer?',
                                                           'Which drink does the user prefer?']


def test_the_cycle_inputs_are_never_modified_on_accept():
    memory_before, suite_before = state()
    before = (asdict(memory_before), asdict(suite_before))
    engine_for(accepted_llm()).consolidate(memory_before, suite_before, 'I prefer tea.')
    assert (asdict(memory_before), asdict(suite_before)) == before


def test_a_rejected_cycle_returns_copies_of_the_original_state():
    memory_before, suite_before = state()
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]], answer=answers_from({}))
    result = engine_for(llm, max_repairs=1).consolidate(memory_before, suite_before, 'I prefer tea.')
    assert not result.accepted and result.attempts == 2
    assert asdict(result.memory) == asdict(memory_before) and result.memory is not memory_before
    assert asdict(result.suite) == asdict(suite_before) and result.suite is not suite_before
    assert result.verification.failed_test_ids == ['0', '1', '2']


def test_repairs_are_tried_before_rejecting():
    def answer_coffee_then_tea(query, items):
        contents = ' '.join(item['content'] for item in items).lower()
        if query == 'Where was the user born?':
            return 'Pittsburgh'
        return 'tea' if 'tea' in contents else 'coffee'

    memory_before, suite_before = state()
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]],
                      memory_proposals=[[write('User prefers coffee.')]],
                      repairs=[[write('User prefers tea.')]], answer=answer_coffee_then_tea)
    result = engine_for(llm, max_repairs=2).consolidate(memory_before, suite_before, 'I prefer tea.')
    assert result.accepted and result.attempts == 2
    assert [operation.content for operation in result.operations] == ['User prefers coffee.', 'User prefers tea.']
    assert len(llm.prompts['repair']) == 1


def test_repairs_are_bounded_by_max_repairs():
    memory_before, suite_before = state()
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]], answer=answers_from({}))
    result = engine_for(llm, max_repairs=3).consolidate(memory_before, suite_before, 'evidence')
    assert not result.accepted and result.attempts == 4
    assert len(llm.prompts['repair']) == 3


def test_zero_repairs_means_one_attempt():
    memory_before, suite_before = state()
    llm = ScriptedLLM(answer=answers_from({}))
    result = engine_for(llm, max_repairs=0).consolidate(memory_before, suite_before, 'evidence')
    assert not result.accepted and result.attempts == 1 and llm.prompts['repair'] == []


def test_negative_repairs_are_rejected():
    with pytest.raises(ValueError, match='max_repairs must be nonnegative'):
        engine_for(ScriptedLLM(), max_repairs=-1)


def test_repair_sees_the_answer_memory_gave_but_no_gold():
    memory_before, suite_before = state()
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]],
                      answer=answers_from({}))
    engine_for(llm, max_repairs=1).consolidate(memory_before, suite_before, 'evidence')
    payload = json.loads(llm.prompts['repair'][0])
    assert payload['failed_tests'] == [{'test_id': '0', 'query': 'Where was the user born?',
                                       'predicted_answer': 'unknown'},
                                      {'test_id': '1', 'query': 'Which drink does the user prefer?',
                                       'predicted_answer': 'unknown'},
                                      {'test_id': '2', 'query': 'Which drink does the user prefer?',
                                       'predicted_answer': 'unknown'}]
    said = llm.prompts['repair'][0]
    # The candidate memory is visible (repair patches it); the reference answers and the judge's
    # reasoning are not, so a repair cannot copy gold into memory to pass.
    assert all(set(failure) == {'test_id', 'query', 'predicted_answer'} for failure in payload['failed_tests'])
    assert 'expected_answer' not in said and 'reason' not in said


def test_repair_is_offered_previous_versions_of_changed_items():
    memory_before, suite_before = state()
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]],
                      memory_proposals=[[revise('0', 'User was born in Boston.')]], answer=answers_from({}))
    engine_for(llm, max_repairs=1).consolidate(memory_before, suite_before, 'evidence')
    payload = json.loads(llm.prompts['repair'][0])
    assert payload['previous_versions_of_changed_items'] == [
        {'id': '0', 'content': 'User was born in Pittsburgh.', 'source_ids': []}]


def test_repairs_cannot_weaken_the_frozen_suite():
    memory_before, suite_before = state()
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]], answer=answers_from({}))
    engine_for(llm, max_repairs=2).consolidate(memory_before, suite_before, 'evidence')
    # Every verification ran the same three probes: the repair loop never re-generates or shrinks it.
    assert len(llm.prompts['judge']) == 9 and len(llm.prompts['solver']) == 9
    assert len(llm.prompts['tests']) == 1


def test_a_malformed_memory_proposal_is_an_empty_patch_not_an_error():
    memory_before, suite_before = state()
    llm = accepted_llm(memory_proposals=[[prune('no-such-id')]])
    result = engine_for(llm, max_repairs=0).consolidate(memory_before, suite_before, 'evidence')
    assert not result.accepted and result.attempts == 1
    assert [item.content for item in result.memory.items] == ['User was born in Pittsburgh.']


def test_a_malformed_probe_proposal_does_not_abort_the_cycle():
    # The cumulative suite rejects anything but WRITE; a bad probe used to raise out of consolidate().
    memory_before, suite_before = state()
    llm = accepted_llm(test_proposals=[[probe('Which drink does the user prefer?', 'tea', action='PRUNE',
                                              target_id='0')]])
    result = engine_for(llm).consolidate(memory_before, suite_before, 'I prefer tea.')
    assert result.accepted
    assert [test.query for test in result.suite.tests] == ['Where was the user born?',
                                                           'Which drink does the user prefer?']


def test_a_truncated_solver_batch_is_never_an_acceptance():
    # The headline bug: zip() dropped the unchecked probes and all(checked) held, so a solver that
    # answered nothing produced accepted=True with 0 probes verified.
    memory_before, suite_before = state()
    llm = accepted_llm(solver_batch=lambda prompts: [])
    with pytest.raises(BatchLengthError, match='solver returned 0 outputs'):
        engine_for(llm).consolidate(memory_before, suite_before, 'I prefer tea.')


def test_a_truncated_judge_batch_is_never_an_acceptance():
    memory_before, suite_before = state()
    llm = accepted_llm(judge_batch=lambda prompts: prompts and [])
    with pytest.raises(BatchLengthError, match='judge returned 0 outputs'):
        engine_for(llm).consolidate(memory_before, suite_before, 'I prefer tea.')


def test_version_is_unchanged_when_no_operations_are_proposed():
    memory_before = memory('User was born in Pittsburgh.', 'User prefers tea.')
    suite_before = suite(('Where was the user born?', 'Pittsburgh'), ('Which drink does the user prefer?', 'tea'))
    result = engine_for(ScriptedLLM(answer=answer_from_memory), max_repairs=0).consolidate(
        memory_before, suite_before, 'nothing new')
    assert result.accepted and result.operations == []
    assert result.memory.version == memory_before.version


def test_save_run_writes_a_trace_of_an_accepted_cycle(tmp_path):
    memory_before, suite_before = state()
    result = engine_for(accepted_llm()).consolidate(memory_before, suite_before, 'I prefer tea.',
                                                   save_run=True, output_dir=tmp_path / 'runs')
    trace = json.loads(Path(result.run_path).read_text())
    assert trace['status'] == 'accepted' and trace['attempts'] == 1
    assert trace['evidence'] == 'I prefer tea.'
    assert [event['event'] for event in trace['events']] == ['regression_proposal', 'expanded_suite',
                                                            'memory_proposal', 'candidate', 'verification']
    assert trace['final_memory']['items'][1]['content'] == 'User prefers tea.'
    assert Path(result.run_path).parent == (tmp_path / 'runs')


def test_save_run_marks_a_rejected_cycle(tmp_path):
    memory_before, suite_before = state()
    result = engine_for(ScriptedLLM(answer=answers_from({})), max_repairs=0).consolidate(
        memory_before, suite_before, 'evidence', save_run=True, output_dir=tmp_path)
    trace = json.loads(Path(result.run_path).read_text())
    assert trace['status'] == 'rejected'
    assert trace['events'][-1]['event'] == 'verification' and not trace['events'][-1]['passed']


def test_save_run_records_a_malformed_probe_proposal(tmp_path):
    memory_before, suite_before = state()
    llm = accepted_llm(test_proposals=[[probe('q?', 'a', action='PRUNE', target_id='0')]])
    result = engine_for(llm).consolidate(memory_before, suite_before, 'evidence', save_run=True,
                                         output_dir=tmp_path)
    trace = json.loads(Path(result.run_path).read_text())
    assert trace['events'][0]['event'] == 'invalid_regression_proposal'
    assert 'only permits WRITE' in trace['events'][0]['error']


def test_save_run_records_and_reraises_an_error(tmp_path):
    memory_before, suite_before = state()
    with pytest.raises(BatchLengthError):
        engine_for(accepted_llm(solver_batch=lambda prompts: [])).consolidate(
            memory_before, suite_before, 'evidence', save_run=True, output_dir=tmp_path)
    (path,) = list(tmp_path.glob('run-*.json'))
    trace = json.loads(path.read_text())
    assert trace['status'] == 'error' and trace['error']['type'] == 'BatchLengthError'


def test_trace_names_are_unique(tmp_path):
    memory_before, suite_before = state()
    engine = engine_for(ScriptedLLM(answer=answer_from_memory), max_repairs=0)
    first = engine.consolidate(memory_before, suite_before, 'a', save_run=True, output_dir=tmp_path)
    second = engine.consolidate(memory_before, suite_before, 'a', save_run=True, output_dir=tmp_path)
    assert first.run_path != second.run_path
    assert len(list(tmp_path.glob('run-*.json'))) == 2


def test_traces_hold_unicode(tmp_path):
    memory_before, suite_before = state()
    engine_for(ScriptedLLM(answer=answer_from_memory), max_repairs=0).consolidate(
        memory_before, suite_before, 'User mieszka w Gdańsku — 好', save_run=True, output_dir=tmp_path)
    (path,) = list(tmp_path.glob('run-*.json'))
    assert 'Gdańsku — 好' in path.read_text(encoding='utf-8')
