"""Memory operations apply all-or-nothing, never touch the caller's state, and are validated."""
import json
from dataclasses import asdict

import pytest

from helpers import ScriptedLLM, memory, prune, revise, write

from regmem.proposer import Consolidator, apply_memory_operations, validate_memory_operations


def test_write_appends_with_the_next_id_and_bumps_the_version():
    state = memory('User prefers tea.', version=3)
    result = apply_memory_operations(state, [write('User lives in Gdansk.')])
    assert [(item.id, item.content) for item in result.items] == [('0', 'User prefers tea.'),
                                                                ('1', 'User lives in Gdansk.')]
    assert result.version == 4 and result.next_id == 2


def test_write_requires_content():
    with pytest.raises(ValueError, match='WRITE requires content'):
        apply_memory_operations(memory(), [write('   ')])


def test_write_requires_a_null_target():
    with pytest.raises(ValueError, match='WRITE requires content and no target_id'):
        apply_memory_operations(memory('a'), [write('b', target_id='0')])


def test_revise_replaces_content_in_place():
    result = apply_memory_operations(memory('lives in Sopot', 'prefers tea'), [revise('0', 'lives in Gdansk')])
    assert [(item.id, item.content) for item in result.items] == [('0', 'lives in Gdansk'), ('1', 'prefers tea')]


def test_revise_to_identical_content_changes_nothing_but_the_version():
    state = memory('prefers tea', version=5)
    result = apply_memory_operations(state, [revise('0', 'prefers tea')])
    assert [item.content for item in result.items] == ['prefers tea']
    assert result.version == 6  # the README promises one increment per accepted cycle with operations


def test_revise_rejects_an_unknown_target():
    with pytest.raises(ValueError, match='Unknown memory target_id'):
        apply_memory_operations(memory('a'), [revise('9', 'b')])


def test_revise_rejects_blank_content():
    with pytest.raises(ValueError, match='REVISE requires nonempty content'):
        apply_memory_operations(memory('a'), [revise('0', ' ')])


def test_prune_removes_the_item():
    result = apply_memory_operations(memory('a', 'b'), [prune('0')])
    assert [(item.id, item.content) for item in result.items] == [('1', 'b')]


def test_prune_requires_a_null_content():
    with pytest.raises(ValueError, match='PRUNE requires null content'):
        apply_memory_operations(memory('a'), [prune('0', content='b')])


def test_prune_rejects_an_unknown_target():
    with pytest.raises(ValueError, match='Unknown memory target_id'):
        apply_memory_operations(memory('a'), [prune('7')])


def test_duplicate_ids_in_the_state_are_rejected():
    state = memory('a', 'b')
    state.items[1].id = '0'
    with pytest.raises(ValueError, match='Duplicate memory IDs'):
        apply_memory_operations(state, [])


def test_a_write_onto_an_existing_id_is_rejected():
    with pytest.raises(ValueError, match='next_id collides'):
        apply_memory_operations(memory('a', next_id=0), [write('b')])


def test_operations_apply_in_order():
    result = apply_memory_operations(memory(), [write('first'), revise('0', 'second'), write('third')])
    assert [(item.id, item.content) for item in result.items] == [('0', 'second'), ('1', 'third')]
    assert result.next_id == 2


def test_an_empty_patch_is_a_no_op():
    state = memory('a', version=2)
    result = apply_memory_operations(state, [])
    assert asdict(result) == asdict(state)


def test_the_caller_state_is_never_modified():
    state = memory('a', version=2)
    before = asdict(state)
    apply_memory_operations(state, [write('b'), revise('0', 'c'), prune('1')])
    assert asdict(state) == before


def test_validate_returns_none_for_a_valid_patch():
    assert validate_memory_operations(memory('a'), [write('b')]) is None


def test_validate_raises_and_leaves_the_state_alone():
    state = memory('a')
    before = asdict(state)
    with pytest.raises(ValueError, match='Unknown memory target_id'):
        validate_memory_operations(state, [write('b'), prune('4')])
    assert asdict(state) == before


def test_consolidator_returns_the_proposed_operations():
    llm = ScriptedLLM(memory_proposals=[[write('User prefers tea.')]])
    operations = Consolidator(llm).propose(memory(), 'I prefer tea.')
    assert [operation.action for operation in operations] == ['WRITE']
    assert operations[0].content == 'User prefers tea.'


def test_consolidator_rejects_a_proposal_memory_cannot_apply():
    llm = ScriptedLLM(memory_proposals=[[prune('no-such-id')]])
    with pytest.raises(ValueError, match='Unknown memory target_id'):
        Consolidator(llm).propose(memory(), 'I prefer tea.')


def test_consolidator_does_not_modify_the_memory_it_was_given():
    llm = ScriptedLLM(memory_proposals=[[write('User prefers tea.'), prune('0')]])
    state = memory('User lives in Sopot.', version=4)
    before = asdict(state)
    Consolidator(llm).propose(state, 'I prefer tea.')
    assert asdict(state) == before


def test_propose_prompt_carries_memory_and_evidence():
    llm = ScriptedLLM(memory_proposals=[[]])
    Consolidator(llm).propose(memory('User lives in Sopot.'), 'I prefer tea without milk.')
    payload = json.loads(llm.prompts['propose'][0])
    assert payload['new_evidence'] == 'I prefer tea without milk.'
    assert payload['memory'] == [{'id': '0', 'content': 'User lives in Sopot.', 'source_ids': []}]
    assert 'task' not in payload


def test_repair_prompt_carries_failures_without_gold_answers():
    llm = ScriptedLLM(repairs=[[]])
    failures = [{'test_id': '1', 'query': 'Which drink does the user prefer?', 'predicted_answer': 'coffee'}]
    previous = [{'id': '0', 'content': 'User lives in Sopot.', 'source_ids': []}]
    Consolidator(llm).repair(memory('User prefers coffee.'), 'I prefer tea.', failures, previous)
    payload = json.loads(llm.prompts['repair'][0])
    assert payload['failed_tests'] == failures
    assert payload['previous_versions_of_changed_items'] == previous
    assert set(payload['failed_tests'][0]) == {'test_id', 'query', 'predicted_answer'}
    assert 'task' in payload and 'Repair this candidate' in payload['task']
