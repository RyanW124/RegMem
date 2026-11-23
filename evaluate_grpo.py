"""Paired evaluation of the two RegMem adapters on fixed development transitions."""
import argparse
import sys
from pathlib import Path

from regmem import Consolidator, Verifier
from regmem.grpo_data import (memory_from_dict, proposal_session, read_rows,
                              recoverable_failure, suite_from_dict)
from regmem.proposer import apply_memory_operations, repair_context
from regmem.eval_metrics import write_comparison


def apply_or_keep(memory, propose):
    """Match the engine's treatment of malformed model actions."""
    try:
        return apply_memory_operations(memory, propose())
    except ValueError:
        return memory


def repair_trial(policy, original, candidate, suite, first, verifier, attempts):
    check = first
    new_failures = set()
    for _ in range(attempts):
        if check.passed:
            break
        failures, previous = repair_context(original, candidate, suite, check)
        candidate = apply_or_keep(candidate, lambda: policy.repair(candidate, failures, previous))
        following = verifier.verify(candidate, suite)
        new_failures.update(set(following.failed_test_ids) - set(check.failed_test_ids))
        check = following
    return {'accepted': check.passed, 'new_regressions': len(new_failures)}


def transition_rows(rows, llm, proposal_llm, repair_llm, attempts):
    verifier = Verifier(llm)
    for row in rows:
        original, suite = memory_from_dict(row['memory']), suite_from_dict(row['suite'])
        for proposer, model in [('base', llm), ('initial', proposal_llm)]:
            policy = Consolidator(model)
            candidate = apply_or_keep(original, lambda: policy.propose(original, proposal_session(row)))
            first = verifier.verify(candidate, suite)
            recoverable = None if first.passed else recoverable_failure(original, candidate, suite, verifier)
            # Both repair policies start at exactly the same candidate, with identical feedback.
            for repairer, model in [('base', llm), ('repair', repair_llm)]:
                arm = ('both' if proposer == 'initial' else 'repair') if repairer == 'repair' else proposer
                result = repair_trial(Consolidator(model), original, candidate, suite, first, verifier, attempts)
                yield {'episode_id': row['episode_id'], 'step': row['step'], 'arm': arm,
                       'initial_accepted': first.passed, 'recoverable': recoverable, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=['transitions'], nargs='?', default='transitions')
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--proposal-adapter', required=True)
    parser.add_argument('--repair-adapter', required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('runs/grpo/data'))
    parser.add_argument('--out', type=Path, default=Path('runs/grpo/eval'))
    parser.add_argument('--max-repairs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = args.out / args.benchmark
    out.mkdir(parents=True, exist_ok=True)
    from transformers import set_seed
    from regmem.llm import LLM
    set_seed(args.seed)
    llm = LLM(args.model, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, temperature=0)
    rows = transition_rows(read_rows(args.data_dir / 'proposal_dev.jsonl'), llm,
                           llm.for_adapter(args.proposal_adapter, 'proposal'),
                           llm.for_adapter(args.repair_adapter, 'repair'), args.max_repairs)
    write_comparison(out, rows)


if __name__ == '__main__':
    if sys.argv[1:2] == ['memconflict']:
        # Translate the former subcommand's output layout and pilot defaults.
        from run_memconflict import main as run_memconflict
        legacy = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        legacy.add_argument('--out', type=Path, default=Path('runs/grpo/eval'))
        legacy.add_argument('--data-dir')
        options, rest = legacy.parse_known_args(sys.argv[2:])
        run_memconflict(['--arm', 'probes', 'regmem', '--compare-adapters', '--instances', '0', '1', '2',
                         '--max-sessions', '20', '--batch-size', '4', '--temperature', '0',
                         '--out', str(options.out / 'memconflict'), *rest])
    else:
        main()
