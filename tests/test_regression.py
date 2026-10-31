"""The cumulative suite only grows, only accepts probes, and never mutates the caller's suite."""
import json
from dataclasses import asdict

import pytest

from helpers import ScriptedLLM, probe, suite

from regmem.regression import RegressionGenerator, apply_regression_operations, validate_regression_operations


def test_write_appends_a_probe_with_the_next_id():
    result = apply_regression_operations(suite(('Where was the user born?', 'Pittsburgh'), version=2),
                                         [probe('Which drink does the user prefer?', 'tea')])
    assert [(test.id, test.query, test.expected_answer) for test in result.tests] == [
        ('0', 'Where was the user born?', 'Pittsburgh'), ('1', 'Which drink does the user prefer?', 'tea')]
    assert result.version == 3 and result.next_id == 2


def test_the_suite_is_cumulative():
    before = suite(('a?', 'a'), ('b?', 'b'))
    result = apply_regression_operations(before, [probe('c?', 'c')])
    assert [test.query for test in result.tests] == ['a?', 'b?', 'c?']


def test_an_empty_proposal_leaves_the_suite_unchanged():
    state = suite(('a?', 'a'), version=3)
    assert asdict(apply_regression_operations(state, [])) == asdict(state)


@pytest.mark.parametrize('operation', [probe('c?', 'c', action='PRUNE', target_id='0'),
                                       probe('c?', 'c', action='REVISE', target_id='0')])
def test_only_write_is_allowed_in_a_cumulative_suite(operation):
    with pytest.raises(ValueError, match='The cumulative suite only permits WRITE'):
        apply_regression_operations(suite(('a?', 'a')), [operation])


def test_write_requires_a_query_and_an_expected_answer():
    with pytest.raises(ValueError, match='WRITE requires query, expected_answer and no target_id'):
        apply_regression_operations(suite(), [probe('   ', 'c')])
    with pytest.raises(ValueError, match='WRITE requires query, expected_answer and no target_id'):
        apply_regression_operations(suite(), [probe('c?', '  ')])
    with pytest.raises(ValueError, match='WRITE requires query, expected_answer and no target_id'):
        apply_regression_operations(suite(), [probe('c?', 'c', target_id='0')])


def test_duplicate_test_ids_are_rejected():
    state = suite(('a?', 'a'), ('b?', 'b'))
    state.tests[1].id = '0'
    with pytest.raises(ValueError, match='Duplicate regression test IDs'):
        apply_regression_operations(state, [])


def test_a_probe_onto_an_existing_id_is_rejected():
    with pytest.raises(ValueError, match='next_id collides'):
        apply_regression_operations(suite(('a?', 'a'), next_id=0), [probe('b?', 'b')])


def test_the_caller_suite_is_never_modified():
    state = suite(('a?', 'a'), version=1)
    before = asdict(state)
    apply_regression_operations(state, [probe('b?', 'b')])
    assert asdict(state) == before


def test_validate_returns_none_for_a_valid_proposal():
    assert validate_regression_operations(suite(('a?', 'a')), [probe('b?', 'b')]) is None


def test_validate_raises_for_a_proposal_the_suite_rejects():
    with pytest.raises(ValueError, match='only permits WRITE'):
        validate_regression_operations(suite(('a?', 'a')), [probe('b?', 'b', action='PRUNE')])


def test_generator_returns_the_proposed_probes():
    llm = ScriptedLLM(test_proposals=[[probe('Which drink does the user prefer?', 'tea')]])
    operations = RegressionGenerator(llm).propose(suite(), 'I prefer tea.')
    assert [(operation.query, operation.expected_answer) for operation in operations] == [
        ('Which drink does the user prefer?', 'tea')]


def test_generator_rejects_a_probe_the_suite_rejects():
    llm = ScriptedLLM(test_proposals=[[probe('b?', 'b', action='REVISE', target_id='0')]])
    with pytest.raises(ValueError, match='only permits WRITE'):
        RegressionGenerator(llm).propose(suite(('a?', 'a')), 'evidence')


def test_generator_does_not_modify_the_suite_it_was_given():
    llm = ScriptedLLM(test_proposals=[[probe('b?', 'b')]])
    state = suite(('a?', 'a'), version=2)
    before = asdict(state)
    RegressionGenerator(llm).propose(state, 'evidence')
    assert asdict(state) == before


def test_generate_prompt_carries_existing_tests_and_evidence():
    llm = ScriptedLLM(test_proposals=[[]])
    RegressionGenerator(llm).propose(suite(('Where was the user born?', 'Pittsburgh')), 'I prefer tea.')
    payload = json.loads(llm.prompts['tests'][0])
    assert payload['new_evidence'] == 'I prefer tea.'
    assert payload['existing_tests'] == [{'id': '0', 'query': 'Where was the user born?',
                                          'expected_answer': 'Pittsburgh'}]
