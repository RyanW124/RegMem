"""Command line front end: consolidate one evidence batch, verify a memory, inspect run traces.

    python -m regmem consolidate --memory memory.json --suite suite.json --evidence note.txt --save-run
    python -m regmem verify --memory memory.json --suite suite.json
    python -m regmem inspect results/runs/run-*.json

`--model stub` dry-runs the pipeline with the deterministic double in `regmem.stub`; any other
value is a Hugging Face model id and needs the optional model dependencies. Exit codes: 0 when the
cycle was accepted (or every probe passed), 1 when it was rejected (or a probe failed), 2 for a
usage, file or model error.
"""
import argparse
import json
from pathlib import Path
import sys

from .consolidation import ConsolidationEngine
from .persistence import (load_memory, load_suite, memory_to_dict, save_memory, save_suite,
                          suite_to_dict)
from .proposer import Consolidator
from .regression import RegressionGenerator
from .typedefs import ConsolidationResult
from .verifier import Verifier

EXIT_OK, EXIT_FAILED, EXIT_ERROR = 0, 1, 2


class UsageError(Exception):
    """A caller-side mistake (bad path, missing dependency) that must exit 2, not traceback."""


def build_llm(model: str):
    if model == 'stub':
        from .stub import StubLLM
        return StubLLM()
    try:
        from .llm import LLM
    except ImportError as error:
        raise UsageError(f'--model {model} needs the optional model dependencies '
                         f'(pip install -r requirements.txt): {error}') from error
    return LLM(model)


def build_engine(llm, max_repairs: int) -> ConsolidationEngine:
    return ConsolidationEngine(Consolidator(llm), RegressionGenerator(llm), Verifier(llm), max_repairs=max_repairs)


def read_evidence(args) -> str:
    if args.text is not None:
        return args.text
    try:
        return Path(args.evidence).read_text(encoding='utf-8')
    except OSError as error:
        raise UsageError(f'cannot read --evidence {args.evidence}: {error}') from error


def report(result: ConsolidationResult, before: tuple[int, int], written: bool, as_json: bool) -> str:
    memory_before, suite_before = before
    memory_after, suite_after = len(result.memory.items), len(result.suite.tests)
    if as_json:
        return json.dumps({'accepted': result.accepted, 'attempts': result.attempts,
                           'memory_items': [memory_before, memory_after],
                           'suite_tests': [suite_before, suite_after],
                           'version': result.memory.version, 'failed_test_ids': result.verification.failed_test_ids,
                           'pass_rate': result.verification.pass_rate, 'written': written,
                           'run_path': result.run_path, 'memory': memory_to_dict(result.memory),
                           'suite': suite_to_dict(result.suite)}, ensure_ascii=False, indent=2)
    verdict = 'accepted' if result.accepted else 'rejected'
    lines = [f'{verdict} after {result.attempts} verification attempt(s): memory {memory_before} -> {memory_after} '
             f'items (version {result.memory.version}), suite {suite_before} -> {suite_after} tests',
             result.verification.summary()]
    if result.run_path:
        lines.append(f'run trace: {result.run_path}')
    lines.append('wrote the accepted state back' if written
                 else 'left the files untouched' if result.accepted else 'memory and suite unchanged')
    return '\n'.join(lines)


def consolidate(args) -> int:
    memory, suite = load_memory(args.memory), load_suite(args.suite)
    before = (len(memory.items), len(suite.tests))
    engine = build_engine(build_llm(args.model), args.max_repairs)
    result = engine.consolidate(memory, suite, read_evidence(args), save_run=args.save_run,
                                output_dir=args.out_dir)
    written = False
    if result.accepted and args.write:
        save_memory(args.memory, result.memory)
        save_suite(args.suite, result.suite)
        written = True
    print(report(result, before, written, args.json))
    return EXIT_OK if result.accepted else EXIT_FAILED


def verify(args) -> int:
    memory, suite = load_memory(args.memory), load_suite(args.suite)
    verification = Verifier(build_llm(args.model)).verify(memory, suite)
    if args.json:
        print(json.dumps({'passed': verification.passed, 'pass_rate': verification.pass_rate,
                          'failed_test_ids': verification.failed_test_ids,
                          'results': [{'test_id': r.test_id, 'passed': r.passed, 'reason': r.reason,
                                       'expected_answer': r.expected_answer,
                                       'predicted_answer': r.predicted_answer}
                                      for r in verification.results]}, ensure_ascii=False, indent=2))
    else:
        print(verification.summary())
        for result in verification.results:
            if not result.passed:
                print(f'  {result.test_id}: expected {result.expected_answer!r}, '
                      f'answered {result.predicted_answer!r} ({result.reason})')
    return EXIT_OK if verification.passed else EXIT_FAILED


def inspect(args) -> int:
    """Summarize saved traces; anything that is not a trace is reported and skipped."""
    found = 0
    for name in args.paths:
        try:
            trace = json.loads(Path(name).read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            print(f'{name}: cannot read ({error})')
            continue
        if not isinstance(trace, dict) or 'events' not in trace or 'status' not in trace:
            print(f'{name}: not a consolidation trace (skipped)')
            continue
        found += 1
        failed = [result['test_id'] for event in trace['events'] if event.get('event') == 'verification'
                  for result in event.get('results', []) if not result['passed']]
        line = (f"{name}: {trace['status']} attempts={trace.get('attempts', '-')} "
                f"events={len(trace['events'])} memory={len(trace.get('final_memory', {}).get('items', []))} "
                f"suite={len(trace.get('final_suite', {}).get('tests', []))}")
        if trace.get('error'):
            line += f" error={trace['error']['type']}: {trace['error']['message']}"
        if failed:
            line += f' failed probes={", ".join(sorted(set(failed)))}'
        print(line)
    return EXIT_OK if found or args.allow_missing else EXIT_ERROR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='regmem', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    run = commands.add_parser('consolidate', help='run one consolidation cycle against an evidence batch')
    run.add_argument('--memory', required=True, help='memory JSON written by save_memory')
    run.add_argument('--suite', required=True, help='regression suite JSON written by save_suite')
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument('--evidence', help='file holding the new evidence batch')
    source.add_argument('--text', help='the evidence batch itself')
    run.add_argument('--model', default='stub', help="'stub' (default, no model needed) or a Hugging Face id")
    run.add_argument('--max-repairs', type=int, default=2)
    run.add_argument('--out-dir', default='results/runs', help='where --save-run writes traces')
    run.add_argument('--save-run', action='store_true', help='write a JSON trace of the cycle')
    run.add_argument('--write', action='store_true', help='save an accepted memory and suite back to their files')
    run.add_argument('--json', action='store_true', help='print the result as JSON')
    run.set_defaults(handler=consolidate)

    check = commands.add_parser('verify', help='re-run the suite against a saved memory')
    check.add_argument('--memory', required=True)
    check.add_argument('--suite', required=True)
    check.add_argument('--model', default='stub')
    check.add_argument('--json', action='store_true')
    check.set_defaults(handler=verify)

    traces = commands.add_parser('inspect', help='summarize saved run traces')
    traces.add_argument('paths', nargs='+')
    traces.add_argument('--allow-missing', action='store_true',
                        help='exit 0 even when nothing readable was found')
    traces.set_defaults(handler=inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except UsageError as error:
        print(f'regmem: error: {error}', file=sys.stderr)
        return EXIT_ERROR
    except (OSError, ValueError) as error:
        # ValueError covers malformed model output (BatchLengthError) and invalid proposals.
        print(f'regmem: error: {type(error).__name__}: {error}', file=sys.stderr)
        return EXIT_ERROR


if __name__ == '__main__':
    raise SystemExit(main())
