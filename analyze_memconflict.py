"""Midterm figure, summary table and hand-labeling sheets from run_memconflict.py output.

  python analyze_memconflict.py --out results/full [--data data/memconflict/Step4_4.jsonl]

Writes {out}/analysis/:
  midterm.png          probe pass rate, static accuracy and memory size vs. cycle (mean over instances)
  summary.md           protocol answer accuracy by conflict type, cycle outcomes, paired static deltas
  judge_sample.csv     ~100 stratified judge decisions: fill human_pass (1/0), rerun for agreement and kappa
  static_failures.csv  every static miss at its issue point: fill category for the failure taxonomy
Labeled CSVs are never overwritten.
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Hue = method, line style = RegMem reject policy (dataviz reference palette, validated for light mode).
STYLE = {'extract': ('(a) no consolidation', '#2a78d6', '-', 'o'),
         'summarize': ('(b) consolidation only', '#eb6834', '-', 's'),
         'regmem-append': ('(c) RegMem, append on reject', '#1baf7a', '-', '^'),
         'regmem-retire': ('(c) RegMem, retire on reject', '#1baf7a', '--', 'v')}
INK, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7', '#fcfcfb'
CATEGORIES = ('never_captured', 'no_covering_probe', 'compacted_before_point_b', 'overwritten_at_point_b',
              'distractor_confusion', 'solver_or_judge_error')
KINDS = ('static_conflict', 'dynamic_conflict', 'conditional_conflict')


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=3):
    return '–' if value is None else f'{value:.{digits}f}'


def load_runs(out):
    """arm -> instance -> per-cycle rows, in STYLE order."""
    runs = defaultdict(dict)
    for path in sorted((out / 'arms').glob('inst*_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            lines = [json.loads(line) for line in stream]
        if len(lines) > 1:
            runs[lines[0]['arm']][lines[0]['args']['instance']] = lines[1:]
    return {arm: runs[arm] for arm in STYLE if arm in runs}


def probe_texts(out, instance):
    """Shared-suite test id -> (query, expected answer); ids are assigned in generation order."""
    path = out / 'probes' / f'inst{instance}.json'
    if not path.exists():
        return {}, {}
    sessions = json.loads(path.read_text())['sessions']
    flat = [op for ops in sessions for op in ops]
    by_session = {c: [op['query'] for op in ops] for c, ops in enumerate(sessions)}
    return {str(n): (op['query'], op['expected_answer']) for n, op in enumerate(flat)}, by_session


def figure(runs, path):
    panels = [('Probe pass rate\n(shared suite)', lambda row: row.get('probe_pass_rate'), True),
              ('Static-query accuracy\n(issued so far)',
               lambda row: mean(q.get('accuracy') for q in row['static_reask']) if row.get('static_reask') else None, True),
              ('Memory size\n(tokens)', lambda row: row.get('memory_tokens'), False)]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 8.5), sharex=True, facecolor=SURFACE)
    for ax, (label, value, markers) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for arm, by_instance in runs.items():
            name, color, style, marker = STYLE[arm]
            by_cycle = defaultdict(list)
            for rows in by_instance.values():
                for row in rows:
                    if value(row) is not None:
                        by_cycle[row['cycle'] + 1].append(value(row))
            xs = sorted(by_cycle)
            ax.plot(xs, [mean(by_cycle[x]) for x in xs], color=color, linestyle=style, linewidth=2,
                    solid_joinstyle='round', solid_capstyle='round', label=name, marker=marker if markers else None,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_ylabel(label, color=INK)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(AXIS)
    for ax in axes[:2]:
        ax.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel('Consolidation cycle (session)', color=INK)
    instances = max(len(by_instance) for by_instance in runs.values())
    fig.suptitle(f'MemConflict, mean over {instances} instance(s)', color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def paired_delta(a, b, rng, resamples=2000):
    """Mean of a - b over the same static queries, with a bootstrap 95% CI over queries."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    diffs = [a[k] - b[k] for k in keys]
    boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(resamples))
    return mean(diffs), boots[int(0.025 * resamples)], boots[int(0.975 * resamples)], len(diffs)


def summary(runs):
    lines = ["## Answer accuracy at each query's issue point (MemConflict protocol)", '',
             '| arm | static (n) | static conflict recognition | dynamic (n) | conditional (n) |',
             '|---|---|---|---|---|']
    static_scores = {}
    for arm, by_instance in runs.items():
        # question_id restarts in every session, so a query is identified by (instance, cycle, question_id).
        queries = [((inst, row['cycle'], q['question_id']), q) for inst, rows in by_instance.items()
                   for row in rows for q in row.get('queries') or []]
        cells = []
        for kind in KINDS:
            scores = [q['accuracy'] for _, q in queries if q['conflict_type'] == kind and q.get('accuracy') is not None]
            cells.append(f'{fmt(mean(scores))} ({len(scores)})')
        recognition = mean(q.get('conflict_recognition_score') for _, q in queries if q['conflict_type'] == KINDS[0])
        lines.append(f'| {STYLE[arm][0]} | {cells[0]} | {fmt(recognition)} | {cells[1]} | {cells[2]} |')
        static_scores[arm] = {key: q['accuracy'] for key, q in queries
                              if q['conflict_type'] == KINDS[0] and q.get('accuracy') is not None}
    lines += ['', '## Cycle outcomes', '',
              '| arm | instances | cycles | outcomes | eval errors | final memory tokens | final probe pass rate |',
              '|---|---|---|---|---|---|---|']
    for arm, by_instance in runs.items():
        rows = [row for rows in by_instance.values() for row in rows]
        outcomes = ', '.join(f'{status} {count}' for status, count in Counter(row['status'] for row in rows).most_common())
        final_tokens = mean(rows_[-1]['memory_tokens'] for rows_ in by_instance.values())
        final_pass = mean(next((row['probe_pass_rate'] for row in reversed(rows_) if 'probe_pass_rate' in row), None)
                          for rows_ in by_instance.values())
        lines.append(f"| {STYLE[arm][0]} | {len(by_instance)} | {len(rows)} | {outcomes} | "
                     f"{sum('eval_error' in row for row in rows)} | {fmt(final_tokens, 0)} | {fmt(final_pass)} |")
    lines += ['', '## Paired static-accuracy deltas (same queries; bootstrap 95% CI over queries)', '',
              '| comparison | mean delta | 95% CI | n |', '|---|---|---|---|']
    rng = random.Random(0)
    for a in runs:
        for b in runs:
            if list(runs).index(a) > list(runs).index(b):
                result = paired_delta(static_scores[a], static_scores[b], rng)
                if result:
                    lines.append(f'| {STYLE[a][0]} minus {STYLE[b][0]} | {result[0]:+.3f} | '
                                 f'[{result[1]:+.3f}, {result[2]:+.3f}] | {result[3]} |')
    return '\n'.join(lines) + '\n'


def write_csv(path, rows, columns):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def judge_sample(runs, out, path, size=100, seed=0):
    probe_pool, downstream_pool = [], []
    for arm, by_instance in runs.items():
        for inst, rows in by_instance.items():
            texts, _ = probe_texts(out, inst)
            for row in rows:
                base = {'arm': arm, 'instance': inst, 'cycle': row['cycle']}
                for r in row.get('probes') or []:
                    probe_pool.append({**base, 'kind': 'probe', 'item_id': r['test_id'],
                                       'question': texts.get(r['test_id'], ('?',))[0], 'reference': r['expected_answer'],
                                       'prediction': r['predicted_answer'], 'judge_pass': int(r['passed'])})
                for q in row.get('queries') or []:
                    if q.get('accuracy') is not None:
                        downstream_pool.append({**base, 'kind': q['conflict_type'], 'item_id': q['question_id'],
                                                'question': q['question'], 'reference': q['reference'],
                                                'prediction': q['answer'], 'judge_pass': int(q['accuracy'] >= 0.5)})
    rng, picked = random.Random(seed), []
    for pool in (probe_pool, downstream_pool):
        unique = list({(x['question'], x['prediction']): x for x in pool}.values())
        for verdict in (1, 0):
            group = [x for x in unique if x['judge_pass'] == verdict]
            picked += rng.sample(group, min(len(group), size // 4))
    write_csv(path, picked, ['kind', 'arm', 'instance', 'cycle', 'item_id', 'question', 'reference', 'prediction',
                             'judge_pass', 'human_pass'])
    return f'wrote {path} ({len(picked)} rows): fill human_pass with 1/0, then rerun for agreement'


def agreement(path):
    with path.open(encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r['human_pass'].strip() in ('0', '1')]
    if not rows:
        return f'{path.name} has no labels yet: fill human_pass with 1/0'
    lines = ['## Judge agreement with hand labels', '', '| judge | n | agreement | Cohen kappa |', '|---|---|---|---|']
    for name, group in (('RegMem probe judge', [r for r in rows if r['kind'] == 'probe']),
                        ('MemConflict judge', [r for r in rows if r['kind'] != 'probe'])):
        pairs = [(int(r['judge_pass']), int(r['human_pass'])) for r in group]
        if not pairs:
            continue
        observed = mean(j == h for j, h in pairs)
        judge_rate, human_rate = mean(j for j, _ in pairs), mean(h for _, h in pairs)
        expected = judge_rate * human_rate + (1 - judge_rate) * (1 - human_rate)
        kappa = (observed - expected) / (1 - expected) if expected < 1 else None
        lines.append(f'| {name} | {len(pairs)} | {observed:.3f} | {fmt(kappa)} |')
    return '\n'.join(lines) + '\n'


def static_conflicts(instance):
    """Point_B session -> where the true value was stated, the true value and the false one."""
    entries = defaultdict(dict)
    for j, session in enumerate(instance['Full_Session_Chain']):
        for e in session.get('Static_Conflict_Information') or []:
            entries[e['Conflict_ID']][e['Role']] = (j, e.get('Value'))
    info = defaultdict(list)
    for roles in entries.values():
        if 'Point_A' in roles and 'Point_B' in roles:
            info[roles['Point_B'][0]].append({'point_a_session': roles['Point_A'][0], 'true_value': roles['Point_A'][1],
                                              'false_value': roles['Point_B'][1]})
    return info


def static_failures(runs, out, data, path):
    instances = {}
    if data and Path(data).exists():
        wanted = {inst for by_instance in runs.values() for inst in by_instance}
        with open(data, encoding='utf-8') as stream:
            instances = {n: static_conflicts(json.loads(line)) for n, line in enumerate(stream) if n in wanted}
    failures = []
    for arm, by_instance in runs.items():
        for inst, rows in by_instance.items():
            _, probes_by_session = probe_texts(out, inst)
            memory = {row['cycle']: ' '.join(m['content'] for m in row['memory']).lower() for row in rows}
            for row in rows:
                for q in row.get('queries') or []:
                    if q['conflict_type'] != KINDS[0] or (q.get('accuracy') is not None and q['accuracy'] >= 1.0):
                        continue
                    conflicts = instances.get(inst, {}).get(row['cycle'], [])
                    conflict = conflicts[0] if len(conflicts) == 1 else {}
                    true = str(conflict.get('true_value') or '')

                    def present(cycle):  # crude hint, only meaningful for distinctive values (names, places, dates)
                        return '' if len(true) < 4 or cycle not in memory else ('yes' if true.lower() in memory[cycle] else 'no')
                    failures.append({'instance': inst, 'arm': arm, 'cycle': row['cycle'], 'question_id': q['question_id'],
                                     'question': q['question'], 'reference': q['reference'], 'answer': q['answer'],
                                     'accuracy': q.get('accuracy'), 'conflict_recognition': q.get('conflict_recognition_score'),
                                     'point_a_session': conflict.get('point_a_session'), 'true_value': true,
                                     'false_value': conflict.get('false_value'),
                                     'probes_from_point_a_session': ' | '.join(probes_by_session.get(conflict.get('point_a_session'), [])),
                                     'true_value_in_memory_before': present(row['cycle'] - 1),
                                     'true_value_in_memory_after': present(row['cycle']), 'category': ''})
    write_csv(path, failures, list(failures[0]) if failures else ['instance', 'arm', 'category'])
    return f"wrote {path} ({len(failures)} rows): fill category with one of {', '.join(CATEGORIES)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='results/memconflict', help='run directory written by run_memconflict.py')
    parser.add_argument('--data', default='data/memconflict/Step4_4.jsonl', help='adds Point_A/B context to the taxonomy sheet')
    args = parser.parse_args()
    out = Path(args.out)
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f'no arm logs under {out / "arms"}')
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    figure(runs, analysis / 'midterm.png')
    report = summary(runs)
    sample, failures = analysis / 'judge_sample.csv', analysis / 'static_failures.csv'
    notes = [agreement(sample) if sample.exists() else judge_sample(runs, out, sample)]
    notes.append(f'{failures} exists, left untouched' if failures.exists() else static_failures(runs, out, args.data, failures))
    report += '\n' + '\n'.join(n for n in notes if n.startswith('##'))
    (analysis / 'summary.md').write_text(report, encoding='utf-8')
    print(report)
    print('\n'.join(n for n in notes if not n.startswith('##')))
    print(f"wrote {analysis / 'midterm.png'} and {analysis / 'summary.md'}")


if __name__ == '__main__':
    main()
