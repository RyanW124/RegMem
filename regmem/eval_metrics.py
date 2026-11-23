"""Summaries for paired RegMem policy evaluations."""
import json
from statistics import mean


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    report = {}
    for arm in ('base', 'initial', 'repair', 'both'):
        arm_rows = [r for r in rows if r['arm'] == arm]
        if not arm_rows:
            continue
        valid = [r for r in arm_rows if r.get('initial_accepted') is not None]
        failed = [r for r in valid if not r['initial_accepted']]
        witnessed = [r for r in failed if r['recoverable']]
        scores = [q['accuracy'] for r in arm_rows for q in r.get('queries', []) if 'accuracy' in q]
        report[arm] = {
            'transitions': len(arm_rows), 'verified_transitions': len(valid),
            'errors': sum(bool(r.get('error') or r.get('eval_error') or r.get('fallback_error'))
                          for r in arm_rows),
            'first_pass_acceptance': ratio(len(valid) - len(failed), len(valid)),
            'final_acceptance': ratio(sum(r['accepted'] for r in valid), len(valid)),
            'initial_failures': len(failed), 'witnessed_recoverable': len(witnessed),
            'recoverable_failure_fraction': ratio(len(witnessed), len(failed)),
            'repair_success_given_recoverable': ratio(sum(r['accepted'] for r in witnessed), len(witnessed)),
            'repair_success_given_failure': ratio(sum(r['accepted'] for r in failed), len(failed)),
            'new_regressions': sum(r['new_regressions'] for r in valid),
            'graded_queries': len(scores), 'downstream_qa': mean(scores) if scores else None,
        }
    return report


def write_comparison(out, rows):
    completed = []
    with (out / 'comparisons.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            completed.append(row)
    if not completed:
        raise ValueError('No evaluation rows')
    report = summarize(completed)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
