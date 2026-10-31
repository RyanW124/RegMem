"""The CLI consolidates, verifies and inspects, and reports outcomes through its exit code."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from helpers import ScriptedLLM, suite, write

from regmem import __main__ as cli
from regmem.persistence import load_memory, save_memory, save_suite


def answer_from_memory(query, items):
    """Answers the tea probe once memory holds the fact, 'unknown' before."""
    contents = ' '.join(item['content'] for item in items).lower()
    return 'tea' if 'tea' in contents else 'unknown'


@pytest.fixture
def workspace(tmp_path):
    """A memory file, a suite file and an evidence file, plus paths for assertions."""
    paths = {'memory': tmp_path / 'memory.json', 'suite': tmp_path / 'suite.json',
             'evidence': tmp_path / 'note.txt', 'runs': tmp_path / 'runs'}
    from helpers import memory
    save_memory(paths['memory'], memory('User lives in Gdansk.'))
    save_suite(paths['suite'], suite(('Which drink does the user prefer?', 'tea')))
    paths['evidence'].write_text('I prefer tea.', encoding='utf-8')
    return paths


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def stub_accepted(monkeypatch):
    llm = ScriptedLLM(memory_proposals=[[write('User prefers tea.')]], answer=answer_from_memory)
    monkeypatch.setattr(cli, 'build_llm', lambda model: llm)
    return llm


def test_consolidate_accepts_and_reports_counts(workspace, capsys, monkeypatch):
    stub_accepted(monkeypatch)
    code, out, err = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                         str(workspace['suite']), '--evidence', str(workspace['evidence']))
    assert code == cli.EXIT_OK and err == ''
    assert 'accepted after 1 verification attempt(s): memory 1 -> 2 items (version 1), suite 1 -> 1 tests' in out
    assert 'probes 1/1 passed' in out
    assert 'left the files untouched' in out


def test_write_persists_an_accepted_cycle(workspace, capsys, monkeypatch):
    stub_accepted(monkeypatch)
    code, out, _ = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--evidence', str(workspace['evidence']), '--write')
    assert code == cli.EXIT_OK and 'wrote the accepted state back' in out
    assert [item.content for item in load_memory(workspace['memory']).items] == ['User lives in Gdansk.',
                                                                                'User prefers tea.']


def test_a_rejected_cycle_exits_1_and_leaves_the_files_alone(workspace, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_llm', lambda model: ScriptedLLM(answer=lambda query, items: 'unknown'))
    before = (workspace['memory'].read_bytes(), workspace['suite'].read_bytes())
    code, out, _ = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--evidence', str(workspace['evidence']), '--write',
                       '--max-repairs', '0')
    assert code == cli.EXIT_FAILED and 'rejected after 1 verification attempt(s)' in out
    assert 'memory and suite unchanged' in out
    assert (workspace['memory'].read_bytes(), workspace['suite'].read_bytes()) == before


def test_json_output_is_machine_readable(workspace, capsys, monkeypatch):
    stub_accepted(monkeypatch)
    code, out, _ = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--text', 'I prefer tea.', '--json')
    payload = json.loads(out)
    assert code == cli.EXIT_OK
    assert payload['accepted'] is True and payload['attempts'] == 1 and payload['written'] is False
    assert payload['memory_items'] == [1, 2] and payload['suite_tests'] == [1, 1]
    assert payload['pass_rate'] == 1.0 and payload['failed_test_ids'] == []
    assert payload['memory']['items'][1]['content'] == 'User prefers tea.'
    assert payload['suite']['tests'][0]['query'] == 'Which drink does the user prefer?'


def test_text_and_evidence_are_interchangeable(workspace, capsys, monkeypatch):
    stub_accepted(monkeypatch)
    code, out, _ = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--text', 'inline evidence')
    assert code == cli.EXIT_OK


def test_save_run_prints_the_trace_path(workspace, capsys, monkeypatch):
    stub_accepted(monkeypatch)
    code, out, _ = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--evidence', str(workspace['evidence']),
                       '--save-run', '--out-dir', str(workspace['runs']))
    assert code == cli.EXIT_OK and 'run trace:' in out
    (trace,) = list(workspace['runs'].glob('run-*.json'))
    assert json.loads(trace.read_text())['status'] == 'accepted'


def test_verify_passes_and_fails_with_the_right_exit_code(workspace, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_llm', lambda model: ScriptedLLM(answer=lambda query, items: 'tea'))
    code, out, _ = run(capsys, 'verify', '--memory', str(workspace['memory']), '--suite', str(workspace['suite']))
    assert code == cli.EXIT_OK and out.strip() == 'probes 1/1 passed'

    monkeypatch.setattr(cli, 'build_llm', lambda model: ScriptedLLM(answer=lambda query, items: 'unknown'))
    code, out, _ = run(capsys, 'verify', '--memory', str(workspace['memory']), '--suite', str(workspace['suite']))
    assert code == cli.EXIT_FAILED
    assert 'probes 0/1 passed (failed: 0)' in out
    assert "0: expected 'tea', answered 'unknown' (scripted verdict)" in out


def test_verify_json(workspace, capsys, monkeypatch):
    monkeypatch.setattr(cli, 'build_llm', lambda model: ScriptedLLM(answer=lambda query, items: 'unknown'))
    code, out, _ = run(capsys, 'verify', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--json')
    payload = json.loads(out)
    assert code == cli.EXIT_FAILED and payload['passed'] is False
    assert payload['results'][0] == {'test_id': '0', 'passed': False, 'reason': 'scripted verdict',
                                     'expected_answer': 'tea', 'predicted_answer': 'unknown'}


def test_a_missing_file_is_a_usage_error(workspace, capsys):
    code, out, err = run(capsys, 'verify', '--memory', str(workspace['memory'] / 'absent.json'),
                         '--suite', str(workspace['suite']))
    assert code == cli.EXIT_ERROR and out == '' and 'error:' in err


def test_a_malformed_memory_file_is_reported(workspace, capsys):
    workspace['memory'].write_text('{"kind": "regmem.memory"}', encoding='utf-8')
    code, _, err = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--text', 'e')
    assert code == cli.EXIT_ERROR and 'memory state is missing' in err and 'version' in err


def test_a_negative_max_repairs_is_a_usage_error(workspace, capsys):
    code, _, err = run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite',
                       str(workspace['suite']), '--text', 'e', '--max-repairs', '-1')
    assert code == cli.EXIT_ERROR and 'max_repairs must be nonnegative' in err


def test_inspect_summarizes_traces_and_skips_other_json(workspace, capsys, monkeypatch):
    stub_accepted(monkeypatch)
    run(capsys, 'consolidate', '--memory', str(workspace['memory']), '--suite', str(workspace['suite']),
        '--evidence', str(workspace['evidence']), '--save-run', '--out-dir', str(workspace['runs']))
    (trace,) = list(workspace['runs'].glob('run-*.json'))
    code, out, _ = run(capsys, 'inspect', str(trace), str(workspace['suite']))
    assert code == cli.EXIT_OK
    assert f'{trace}: accepted attempts=1 events=5 memory=2 suite=1' in out
    assert 'not a consolidation trace (skipped)' in out


def test_inspect_errors_when_nothing_can_be_read(workspace, capsys):
    code, out, _ = run(capsys, 'inspect', str(workspace['suite']))
    assert code == cli.EXIT_ERROR and 'skipped' in out
    code, _, _ = run(capsys, 'inspect', str(workspace['suite']), '--allow-missing')
    assert code == cli.EXIT_OK


def test_inspect_reports_an_error_trace(workspace, capsys, tmp_path):
    trace = tmp_path / 'run-error.json'
    trace.write_text(json.dumps({'schema_version': 1, 'status': 'error', 'events': [],
                                 'error': {'type': 'BatchLengthError', 'message': 'solver returned 0 outputs'}}))
    code, out, _ = run(capsys, 'inspect', str(trace))
    assert code == cli.EXIT_OK and 'BatchLengthError: solver returned 0 outputs' in out


def test_the_module_entry_point_prints_help():
    result = subprocess.run([sys.executable, '-m', 'regmem', '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'consolidate' in result.stdout and 'inspect' in result.stdout


def test_the_module_entry_point_runs_a_stub_cycle(workspace):
    result = subprocess.run([sys.executable, '-m', 'regmem', 'consolidate', '--memory', str(workspace['memory']),
                             '--suite', str(workspace['suite']), '--text', 'I prefer tea.', '--json'],
                            capture_output=True, text=True, cwd=str(Path.cwd()))
    assert result.returncode in (cli.EXIT_OK, cli.EXIT_FAILED), result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) >= {'accepted', 'attempts', 'pass_rate', 'failed_test_ids', 'memory', 'suite'}
