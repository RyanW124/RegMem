"""JSON persistence for memory state and regression suites.

    from regmem.persistence import load_memory, load_suite, save_memory, save_suite

    memory, suite = load_memory('memory.json'), load_suite('suite.json')
    if engine.consolidate(memory, suite, evidence).accepted:
        save_memory('memory.json', result.memory)
        save_suite('suite.json', result.suite)

The files are the plain dataclass fields, so they diff and review like the traces do. Loading
validates every field: ids stay unique, versions stay non-negative, and `next_id` never collides
with an existing integer id (the assigner in `proposer`/`regression` numbers new items from it).
Writes go through a temporary file in the destination directory and `os.replace`, so a crashed
or concurrent write cannot leave a half-written memory behind.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .typedefs import DecisionCase, MemoryItem, MemoryState, RegressionSuite

MEMORY_KEY, SUITE_KEY = 'regmem.memory', 'regmem.suite'


def memory_to_dict(state: MemoryState) -> dict:
    """Plain-JSON view of a memory state: version, next_id and the items in order."""
    return {'kind': MEMORY_KEY, 'version': state.version, 'next_id': state.next_id,
            'items': [{'id': item.id, 'content': item.content, 'source_ids': list(item.source_ids)}
                      for item in state.items]}


def suite_to_dict(suite: RegressionSuite) -> dict:
    """Plain-JSON view of a regression suite: version, next_id and the probes in order."""
    return {'kind': SUITE_KEY, 'version': suite.version, 'next_id': suite.next_id,
            'tests': [{'id': test.id, 'query': test.query, 'expected_answer': test.expected_answer}
                      for test in suite.tests]}


def memory_from_dict(data: dict) -> MemoryState:
    """Rebuild a MemoryState, raising ValueError if the payload is not a valid one."""
    _check_kind(data, MEMORY_KEY, 'memory state')
    _check_mapping(data, '', {'kind', 'version', 'next_id', 'items'}, set(), 'memory state')
    version, next_id = _check_counter(data['version'], 'version'), _check_counter(data['next_id'], 'next_id')
    entries = _check_sequence(data['items'], 'items', 'memory state')
    items = [_memory_item(entry, position) for position, entry in enumerate(entries)]
    _check_unique([item.id for item in items], 'memory item')
    _check_next_id(next_id, [item.id for item in items], 'memory item')
    return MemoryState(items=items, version=version, next_id=next_id)


def suite_from_dict(data: dict) -> RegressionSuite:
    """Rebuild a RegressionSuite, raising ValueError if the payload is not a valid one."""
    _check_kind(data, SUITE_KEY, 'regression suite')
    _check_mapping(data, '', {'kind', 'version', 'next_id', 'tests'}, set(), 'regression suite')
    version, next_id = _check_counter(data['version'], 'version'), _check_counter(data['next_id'], 'next_id')
    entries = _check_sequence(data['tests'], 'tests', 'regression suite')
    tests = [_test_case(entry, position) for position, entry in enumerate(entries)]
    _check_unique([test.id for test in tests], 'regression test')
    _check_next_id(next_id, [test.id for test in tests], 'regression test')
    return RegressionSuite(tests=tests, version=version, next_id=next_id)


def save_memory(path: str | Path, state: MemoryState) -> Path:
    """Write the memory state to path atomically and return the path."""
    return _write(path, memory_to_dict(state))


def save_suite(path: str | Path, suite: RegressionSuite) -> Path:
    """Write the regression suite to path atomically and return the path."""
    return _write(path, suite_to_dict(suite))


def load_memory(path: str | Path) -> MemoryState:
    """Read a memory state written by save_memory (ValueError on a malformed file)."""
    return memory_from_dict(_read(path))


def load_suite(path: str | Path) -> RegressionSuite:
    """Read a regression suite written by save_suite (ValueError on a malformed file)."""
    return suite_from_dict(_read(path))


def _read(path: str | Path) -> dict:
    with Path(path).open(encoding='utf-8') as stream:
        try:
            data = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f'{path} is not valid JSON: {error}') from error
    if not isinstance(data, dict):
        raise ValueError(f'{path} holds {type(data).__name__}, expected a JSON object')
    return data


def _write(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.',
                                     suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _memory_item(entry, position: int) -> MemoryItem:
    where = f'items[{position}]'
    _check_mapping(entry, where, {'id', 'content'}, {'source_ids'}, 'memory item')
    content = _check_text(entry['content'], f'{where}.content')
    source_ids = entry.get('source_ids', [])
    _check_sequence(source_ids, f'{where}.source_ids', 'memory item')
    return MemoryItem(_check_text(entry['id'], f'{where}.id'), content,
                      [_check_text(source, f'{where}.source_ids[{n}]')
                       for n, source in enumerate(source_ids)])


def _test_case(entry, position: int) -> DecisionCase:
    where = f'tests[{position}]'
    _check_mapping(entry, where, {'id', 'query', 'expected_answer'}, set(), 'regression test')
    return DecisionCase(_check_text(entry['id'], f'{where}.id'),
                        _check_text(entry['query'], f'{where}.query'),
                        _check_text(entry['expected_answer'], f'{where}.expected_answer'))


def _check_kind(data, expected: str, what: str) -> None:
    # Reported before the field-by-field checks so loading a suite as a memory says so.
    if isinstance(data, dict) and 'kind' in data and data['kind'] != expected:
        raise ValueError(f'{what} has kind {data["kind"]!r}, expected {expected!r}')


def _check_mapping(value, where: str, required: set[str], optional: set[str], what: str) -> None:
    subject = f'{what} {where}'.strip()
    if not isinstance(value, dict):
        raise ValueError(f'{subject} must be a JSON object, got {type(value).__name__}')
    missing, unknown = required - set(value), set(value) - required - optional
    if missing:
        raise ValueError(f'{subject} is missing {", ".join(sorted(missing))}')
    if unknown:
        raise ValueError(f'{subject} has unknown field(s) {", ".join(sorted(unknown))}')


def _check_sequence(value, where: str, what: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f'{what} {where} must be a JSON array, got {type(value).__name__}')
    return value


def _check_text(value, where: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{where} must be a string, got {type(value).__name__}')
    if not value.strip():
        raise ValueError(f'{where} must not be blank')
    return value


def _check_counter(value, where: str) -> int:
    # bool is an int subclass; a JSON `true` is not a version.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{where} must be a non-negative integer, got {value!r}')
    return value


def _check_unique(ids: list[str], what: str) -> None:
    seen = set()
    for item_id in ids:
        if item_id in seen:
            raise ValueError(f'duplicate {what} id {item_id!r}')
        seen.add(item_id)


def _check_next_id(next_id: int, ids: list[str], what: str) -> None:
    # New ids are assigned as str(next_id), str(next_id + 1), ..., so a next_id at or below an
    # existing integer id would make the next WRITE raise 'next_id collides'.
    numeric = [int(item_id) for item_id in ids if item_id.isdigit()]
    if numeric and next_id <= max(numeric):
        raise ValueError(f'next_id {next_id} would collide with existing {what} {max(numeric)}; '
                         f'use {max(numeric) + 1} or higher')
