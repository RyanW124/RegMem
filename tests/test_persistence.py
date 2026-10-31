"""Memory and suites survive a JSON round trip, and a malformed file is refused with a reason."""
from dataclasses import asdict
import json

import pytest

from helpers import memory, suite

from regmem.persistence import (load_memory, load_suite, memory_from_dict, memory_to_dict, save_memory,
                               save_suite, suite_from_dict, suite_to_dict)
from regmem.typedefs import MemoryItem


def test_memory_round_trip_is_lossless():
    before = memory('User lives in Gdańsk.', 'User avoids milk.', version=7)
    before.items[0].source_ids = ['ev-1', 'ev-2']
    assert asdict(memory_from_dict(memory_to_dict(before))) == asdict(before)


def test_suite_round_trip_is_lossless():
    before = suite(('Where was the user born?', 'Pittsburgh'), ('Which drink?', 'tea'), version=3)
    assert asdict(suite_from_dict(suite_to_dict(before))) == asdict(before)


def test_save_and_load_a_memory_file(tmp_path):
    path = tmp_path / 'state' / 'memory.json'
    saved = save_memory(path, memory('User lives in Gdańsk — 好.', version=2))
    assert saved == path and path.exists()
    loaded = load_memory(path)
    assert [item.content for item in loaded.items] == ['User lives in Gdańsk — 好.']
    assert loaded.version == 2 and loaded.next_id == 1


def test_save_and_load_a_suite_file(tmp_path):
    path = tmp_path / 'suite.json'
    save_suite(path, suite(('Where was the user born?', 'Pittsburgh'), version=5))
    loaded = load_suite(path)
    assert [(test.id, test.query, test.expected_answer) for test in loaded.tests] == [
        ('0', 'Where was the user born?', 'Pittsburgh')]
    assert loaded.version == 5 and loaded.next_id == 1


def test_saving_replaces_an_existing_file_without_leaving_temporaries(tmp_path):
    path = tmp_path / 'memory.json'
    save_memory(path, memory('first'))
    save_memory(path, memory('second'))
    assert [item.content for item in load_memory(path).items] == ['second']
    assert [entry.name for entry in tmp_path.iterdir()] == ['memory.json']


def test_a_file_is_written_as_utf8_with_json_indentation(tmp_path):
    path = save_memory(tmp_path / 'memory.json', memory('Gdańsk'))
    text = path.read_text(encoding='utf-8')
    assert 'Gdańsk' in text and text.endswith('\n') and '\n  ' in text


def test_duplicate_memory_ids_are_refused():
    payload = {'kind': 'regmem.memory', 'version': 1, 'next_id': 2,
               'items': [{'id': '0', 'content': 'a'}, {'id': '0', 'content': 'b'}]}
    with pytest.raises(ValueError, match='duplicate memory item id'):
        memory_from_dict(payload)


def test_a_next_id_that_would_collide_is_refused():
    payload = {'kind': 'regmem.memory', 'version': 1, 'next_id': 0,
               'items': [{'id': '0', 'content': 'a'}]}
    with pytest.raises(ValueError, match='next_id 0 would collide with existing memory item 0'):
        memory_from_dict(payload)


def test_non_numeric_ids_do_not_constrain_next_id():
    payload = {'kind': 'regmem.memory', 'version': 0, 'next_id': 0,
               'items': [{'id': 'note-a', 'content': 'a'}]}
    assert memory_from_dict(payload).next_id == 0


@pytest.mark.parametrize('field, value, message', [
    ('version', -1, 'version must be a non-negative integer'),
    ('version', True, 'version must be a non-negative integer'),
    ('version', '1', 'version must be a non-negative integer'),
    ('next_id', None, 'next_id must be a non-negative integer'),
])
def test_counters_are_validated(field, value, message):
    payload = {'kind': 'regmem.memory', 'version': 0, 'next_id': 0, 'items': []}
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        memory_from_dict(payload)


def test_a_missing_field_is_named():
    payload = {'kind': 'regmem.memory', 'version': 0, 'items': []}
    with pytest.raises(ValueError, match='missing next_id'):
        memory_from_dict(payload)


def test_an_unknown_field_is_refused():
    payload = {'kind': 'regmem.memory', 'version': 0, 'next_id': 0, 'items': [], 'extra': 1}
    with pytest.raises(ValueError, match='unknown field'):
        memory_from_dict(payload)


def test_loading_a_suite_as_a_memory_is_refused():
    with pytest.raises(ValueError, match="has kind 'regmem.suite', expected 'regmem.memory'"):
        memory_from_dict(suite_to_dict(suite()))


def test_items_must_be_objects_with_text():
    with pytest.raises(ValueError, match='items must be a JSON array'):
        memory_from_dict({'kind': 'regmem.memory', 'version': 0, 'next_id': 0, 'items': {}})
    with pytest.raises(ValueError, match='items\\[0\\].content must be a string'):
        memory_from_dict({'kind': 'regmem.memory', 'version': 0, 'next_id': 0,
                          'items': [{'id': '0', 'content': 4}]})
    with pytest.raises(ValueError, match='items\\[0\\].content must not be blank'):
        memory_from_dict({'kind': 'regmem.memory', 'version': 0, 'next_id': 0,
                          'items': [{'id': '0', 'content': '   '}]})


def test_source_ids_are_optional_and_validated():
    payload = {'kind': 'regmem.memory', 'version': 0, 'next_id': 1,
               'items': [{'id': '0', 'content': 'a'}]}
    assert memory_from_dict(payload).items == [MemoryItem('0', 'a')]
    with pytest.raises(ValueError, match='source_ids must be a JSON array'):
        memory_from_dict({**payload, 'items': [{'id': '0', 'content': 'a', 'source_ids': 'ev-1'}]})


def test_probes_must_carry_a_query_and_an_answer():
    base = {'kind': 'regmem.suite', 'version': 0, 'next_id': 1}
    with pytest.raises(ValueError, match='tests\\[0\\].query must not be blank'):
        suite_from_dict({**base, 'tests': [{'id': '0', 'query': ' ', 'expected_answer': 'a'}]})
    with pytest.raises(ValueError, match='missing expected_answer'):
        suite_from_dict({**base, 'tests': [{'id': '0', 'query': 'q?'}]})


def test_a_malformed_file_reports_the_path(tmp_path):
    path = tmp_path / 'memory.json'
    path.write_text('{not json')
    with pytest.raises(ValueError, match='memory.json is not valid JSON'):
        load_memory(path)
    path.write_text('[1, 2]')
    with pytest.raises(ValueError, match='holds list, expected a JSON object'):
        load_memory(path)


def test_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_memory(tmp_path / 'absent.json')


def test_saved_state_is_plain_json(tmp_path):
    path = save_suite(tmp_path / 'suite.json', suite(('q?', 'a')))
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload == {'kind': 'regmem.suite', 'version': 0, 'next_id': 1,
                       'tests': [{'id': '0', 'query': 'q?', 'expected_answer': 'a'}]}
