"""Figures, summary tables and hand-labeling sheets from run_memconflict.py output.

  python analyze_memconflict.py --out results/final [--budget 2000] [--data data/memconflict/Step4_4.jsonl]

Writes {out}/analysis/:
  budget.png           answer accuracy vs. memory budget per conflict type (95% CI from resampling users)
  cycles.png           probe pass rate, static accuracy and memory size vs. cycle at --budget
  summary.md           accuracy by budget and arm, paired differences at --budget, cycle outcomes
  judge_sample.csv     ~100 stratified judge decisions at --budget: fill human_pass (1/0), rerun for agreement
  static_failures.csv  every static miss at --budget: fill category for the failure taxonomy
Labeled CSVs are never overwritten. --budget defaults to the budget most arms ran at (0 = no cap).
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

# Hue = method in the reference categorical order (validated together for light mode; aqua and yellow are
# under 3:1 on the surface, so every series also has its own marker and summary.md gives the numbers).
# Line style = the midterm's RegMem reject policies.
STYLE = {'extract': ('(a) append-only', '#2a78d6', '-', 'o'),
         'summarize': ('(b) unverified consolidation', '#eb6834', '-', 's'),
         'regmem': ('(c) RegMem', '#1baf7a', '-', '^'),
         'regmem-append': ('(c) RegMem, append on reject', '#1baf7a', '--', '^'),
         'regmem-retire': ('(c) RegMem, retire on reject', '#1baf7a', ':', 'v'),
         'trustmem': ('(d) transition-local check', '#eda100', '-', 'D')}
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
    """(arm, budget) -> instance -> per-cycle rows, in STYLE order then budget. Midterm logs have no budget (0)."""
    runs = defaultdict(dict)
    for path in sorted((out / 'arms').glob('inst*_*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            lines = [json.loads(line) for line in stream]
        if len(lines) > 1:
            runs[(lines[0]['arm'], lines[0].get('budget', 0))][lines[0]['args']['instance']] = lines[1:]
    return {key: runs[key] for key in sorted(runs, key=lambda k: (list(STYLE).index(k[0]), k[1]))}


def budget_name(budget):
    return 'no cap' if not budget else f'{budget / 1000:g}K'


def at_budget(runs, budget):
    """arm -> instance -> rows for one budget."""
    return {arm: by_instance for (arm, b), by_instance in runs.items() if b == budget}


def scores_by_user(by_instance, kind):
    """instance -> answer accuracies of this kind at each query's issue point."""
    return {inst: [q['accuracy'] for row in rows for q in row.get('queries') or []
                   if q['conflict_type'] == kind and q.get('accuracy') is not None]
            for inst, rows in by_instance.items()}


def user_bootstrap(values_by_user, rng, resamples=2000):
    """Pooled mean over every value, with a 95% CI from resampling users (each user keeps all their values)."""
    users = [u for u, values in values_by_user.items() if values]
    if not users:
        return None
    pooled = lambda picked: mean([v for u in picked for v in values_by_user[u]])
    boots = sorted(pooled(rng.choices(users, k=len(users))) for _ in range(resamples))
    return pooled(users), boots[int(0.025 * resamples)], boots[int(0.975 * resamples)], len(users)


def paired_by_user(a, b, kind):
    """instance -> accuracy differences a - b on the same queries (both arms answer every query)."""
    diffs = {}
    for inst in set(a) & set(b):
        scores = [{(row['cycle'], q['question_id']): q['accuracy'] for row in rows for q in row.get('queries') or []
                   if q['conflict_type'] == kind and q.get('accuracy') is not None} for rows in (a[inst], b[inst])]
        diffs[inst] = [scores[0][k] - scores[1][k] for k in sorted(set(scores[0]) & set(scores[1]))]
    return diffs


def probe_texts(out, instance):
    """Shared-suite test id -> (query, expected answer); ids are assigned in generation order."""
    path = out / 'probes' / f'inst{instance}.json'
    if not path.exists():
        return {}, {}
    sessions = json.loads(path.read_text())['sessions']
    flat = [op for ops in sessions for op in ops]
    by_session = {c: [op['query'] for op in ops] for c, ops in enumerate(sessions)}
    return {str(n): (op['query'], op['expected_answer']) for n, op in enumerate(flat)}, by_session


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)


def budget_figure(runs, path, rng):
    """Accuracy vs. memory budget, one panel per conflict type; budgets double, so they sit evenly spaced."""
    budgets = sorted({b for _, b in runs}, key=lambda b: (b == 0, b))
    arms = list(dict.fromkeys(arm for arm, _ in runs))
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True, facecolor=SURFACE)
    top = 0.0
    for ax, kind in zip(axes, ('static_conflict', 'conditional_conflict', 'dynamic_conflict')):
        style_axes(ax)
        for n, arm in enumerate(arms):
            name, color, style, marker = STYLE[arm]
            points = [(budgets.index(b), user_bootstrap(scores_by_user(runs[(arm, b)], kind), rng))
                      for b in budgets if (arm, b) in runs]
            points = [(x + (n - (len(arms) - 1) / 2) * 0.07, ci) for x, ci in points if ci]  # dodge the CI bars
            if not points:
                continue
            xs, ys = [x for x, _ in points], [ci[0] for _, ci in points]
            errors = [[ci[0] - ci[1] for _, ci in points], [ci[2] - ci[0] for _, ci in points]]
            top = max(top, max(ci[2] for _, ci in points))
            ax.errorbar(xs, ys, yerr=errors, color=color, linestyle=style, linewidth=2, elinewidth=1, capsize=3,
                        marker=marker, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        ax.set_title(kind.replace('_conflict', '').capitalize() + ' questions', color=INK)
        ax.set_xticks(range(len(budgets)), [budget_name(b) for b in budgets])
        ax.set_xlim(-0.5, len(budgets) - 0.5)
        ax.set_xlabel('Memory budget (tokens)', color=INK)
    axes[0].set_ylabel('Answer accuracy', color=INK)
    axes[0].set_ylim(0, min(1.0, top + 0.05))
    instances = max(len(by_instance) for by_instance in runs.values())
    fig.suptitle(f'MemConflict, {instances} users, 95% CI from resampling users', color=INK)
    legend = {}  # a panel can be empty (no questions of that kind yet), so collect series from every panel
    for ax in axes:
        legend.update(zip(*reversed(ax.get_legend_handles_labels())))
    fig.legend(list(legend.values()), list(legend), loc='upper center', bbox_to_anchor=(0.5, 0.92),
               ncol=max(1, len(legend)), frameon=False, labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def figure(runs, path, budget=0):
    panels = [('Probe pass rate\n(shared suite)', lambda row: row.get('probe_pass_rate'), True),
              ('Static-query accuracy\n(issued so far)',
               lambda row: mean(q.get('accuracy') for q in row['static_reask']) if row.get('static_reask') else None, True),
              ('Memory size\n(tokens)', lambda row: row.get('memory_tokens'), False)]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 8.5), sharex=True, facecolor=SURFACE)
    for ax, (label, value, markers) in zip(axes, panels):
        style_axes(ax)
        for arm, by_instance in runs.items():
            name, color, style, marker = STYLE[arm]
            by_cycle = defaultdict(list)
            for rows in by_instance.values():
                for row in rows:
                    if value(row) is not None:
                        by_cycle[row['cycle'] + 1].append(value(row))
            # Instances differ in session count; plot only cycles every instance reached so the mean
            # does not jump when an instance drops out.
            xs = [x for x in sorted(by_cycle) if len(by_cycle[x]) == len(by_instance)]
            ax.plot(xs, [mean(by_cycle[x]) for x in xs], color=color, linestyle=style, linewidth=2,
                    solid_joinstyle='round', solid_capstyle='round', label=name, marker=marker if markers else None,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.set_ylabel(label, color=INK)
    for ax in axes[:2]:
        ax.set_ylim(-0.02, 1.02)
    axes[-1].set_xlabel('Consolidation cycle (session)', color=INK)
    instances = max(len(by_instance) for by_instance in runs.values())
    fig.suptitle(f'MemConflict, mean over {instances} instance(s), memory budget {budget_name(budget)}', color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def ci_cell(result, signed=False):
    if not result:
        return '–'
    f = '{:+.3f}' if signed else '{:.3f}'
    return f'{f.format(result[0])} [{f.format(result[1])}, {f.format(result[2])}]'


def summary(runs, budget, rng):
    kinds = ('static_conflict', 'conditional_conflict', 'dynamic_conflict')
    lines = ["## Answer accuracy at each query's issue point (MemConflict protocol)", '',
             'Pooled over questions; 95% CI from resampling users.', '',
             '| budget | arm | users | static | conditional | dynamic | all | static conflict recognition |',
             '|---|---|---|---|---|---|---|---|']
    for (arm, b), by_instance in sorted(runs.items(), key=lambda kv: (kv[0][1] == 0, kv[0][1])):
        cells = [ci_cell(user_bootstrap(scores_by_user(by_instance, kind), rng)) for kind in kinds]
        every = {inst: [v for kind in kinds for v in scores_by_user(by_instance, kind)[inst]] for inst in by_instance}
        recognition = mean(q.get('conflict_recognition_score') for rows in by_instance.values() for row in rows
                           for q in row.get('queries') or [] if q['conflict_type'] == 'static_conflict')
        lines.append(f'| {budget_name(b)} | {STYLE[arm][0]} | {len(by_instance)} | {" | ".join(cells)} | '
                     f'{ci_cell(user_bootstrap(every, rng))} | {fmt(recognition)} |')
    chosen = at_budget(runs, budget)
    lines += ['', f'## Paired differences at {budget_name(budget)} (same questions; 95% CI from resampling users)', '',
              '| comparison | static | conditional | dynamic |', '|---|---|---|---|']
    for a in ('regmem', 'regmem-append'):
        for b in chosen:
            if a in chosen and b != a and not b.startswith('regmem'):
                cells = [ci_cell(user_bootstrap(paired_by_user(chosen[a], chosen[b], kind), rng), signed=True)
                         for kind in kinds]
                lines.append(f'| {STYLE[a][0]} minus {STYLE[b][0]} | {" | ".join(cells)} |')
    lines += ['', f'## Cycle outcomes at {budget_name(budget)}', '',
              'The suite is never shown to the answering model, so its size is reported apart from memory.', '',
              '| arm | users | cycles | outcomes | eval errors | final memory tokens | final suite size '
              '| final probe pass rate |', '|---|---|---|---|---|---|---|---|']
    for arm, by_instance in chosen.items():
        rows = [row for rows in by_instance.values() for row in rows]
        outcomes = ', '.join(f'{status} {count}' for status, count in Counter(row['status'] for row in rows).most_common())
        final = [rows_[-1] for rows_ in by_instance.values()]
        final_pass = mean(next((row['probe_pass_rate'] for row in reversed(rows_) if 'probe_pass_rate' in row), None)
                          for rows_ in by_instance.values())
        lines.append(f"| {STYLE[arm][0]} | {len(by_instance)} | {len(rows)} | {outcomes} | "
                     f"{sum('eval_error' in row for row in rows)} | {fmt(mean(r['memory_tokens'] for r in final), 0)} | "
                     f"{fmt(mean(r['gate_size'] for r in final), 1)} | {fmt(final_pass)} |")
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
    parser.add_argument('--budget', type=int, help='budget for the paired table, cycle figure and labeling sheets')
    args = parser.parse_args()
    out = Path(args.out)
    runs = load_runs(out)
    if not runs:
        raise SystemExit(f'no arm logs under {out / "arms"}')
    budget = args.budget
    if budget is None:  # the budget most arms ran at, preferring a real cap
        budget = max({b for _, b in runs}, key=lambda b: (sum(key[1] == b for key in runs), b != 0, -b))
    chosen, rng = at_budget(runs, budget), random.Random(0)
    if not chosen:
        raise SystemExit(f'no runs at budget {budget}')
    analysis = out / 'analysis'
    analysis.mkdir(exist_ok=True)
    budget_figure(runs, analysis / 'budget.png', rng)
    figure(chosen, analysis / 'cycles.png', budget)
    report = summary(runs, budget, rng)
    sample, failures = analysis / 'judge_sample.csv', analysis / 'static_failures.csv'
    notes = [agreement(sample) if sample.exists() else judge_sample(chosen, out, sample)]
    notes.append(f'{failures} exists, left untouched' if failures.exists() else static_failures(chosen, out, args.data, failures))
    report += '\n' + '\n'.join(n for n in notes if n.startswith('##'))
    (analysis / 'summary.md').write_text(report, encoding='utf-8')
    print(report)
    print('\n'.join(n for n in notes if not n.startswith('##')))
    print(f"wrote {analysis / 'budget.png'}, {analysis / 'cycles.png'} and {analysis / 'summary.md'}")


if __name__ == '__main__':
    main()
