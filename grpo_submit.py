"""Submit unfinished RegMem pilot stages, reconnecting to jobs already in Slurm."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


STAGES = ('prepare', 'proposal', 'mine', 'repair', 'transitions')


def has_data(path):
    return path.is_file() and path.stat().st_size > 0


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def finished(root, stage):
    """The final reports are written after the stage's outputs, including in older runs."""
    data = root / 'data'
    if stage in ('prepare', 'mine'):
        role = 'proposal' if stage == 'prepare' else 'repair'
        report = read_json(data / ('episode_splits.json' if stage == 'prepare' else 'repair_mining.json'))
        if not report:
            return False
        for split in ('train', 'dev'):
            count = (report.get('rows', {}).get(split, 0) if stage == 'prepare'
                     else sum(report.get('splits', {}).get(split, {}).get('repair_rows', {}).values()))
            path = data / f'{role}_{split}.jsonl'
            if count <= 0 or not has_data(path):
                return False
            with path.open('rb') as rows:
                if sum(1 for line in rows if line.strip()) != count:
                    return False
        return stage == 'prepare' or (data / 'repair_failures_dev.jsonl').is_file()
    if stage in ('proposal', 'repair'):
        out = root / stage
        state = read_json(out / 'trainer_state.json') or {}
        return (state.get('global_step', 0) > 0 and state['global_step'] >= state.get('max_steps', float('inf'))
                and bool(read_json(out / 'train_results.json')) and bool(read_json(out / 'adapter_config.json'))
                and has_data(out / 'adapter_model.safetensors'))
    out = root / 'eval' / stage
    metrics = read_json(out / 'metrics.json') or {}
    return (has_data(out / 'comparisons.jsonl') and
            all(metrics.get(arm, {}).get('transitions', 0) > 0 for arm in ('base', 'initial', 'repair', 'both')))


def slurm(*args):
    return subprocess.check_output(args, text=True).strip()


def submit(root, script, run=slurm):
    # Slurm comments survive a login-shell exit, even between sbatch and its reply.
    tag = 'regmem-' + sha256(os.fsencode(root.resolve())).hexdigest()[:20]
    queue = run('squeue', '--me', '--noheader', '--format=%i|%j|%k|%r|%T|%Z')
    jobs = {}
    for line in queue.splitlines():
        job, name, comment, reason, state, cwd = (part.strip() for part in line.split('|', 5))
        if reason == 'DependencyNeverSatisfied' or state in ('FAILED', 'CANCELLED', 'TIMEOUT', 'COMPLETED'):
            continue
        if (name.startswith('grpo-') and name[5:] in STAGES and comment in ('', '(null)', 'N/A')
                and Path(cwd).resolve() == script.parent.resolve()):
            raise SystemExit(f'Legacy job {job} ({name}) is still in the queue. Let the old submission finish '
                             'or resolve its blocked jobs before resubmitting from this checkout.')
        if comment.startswith(tag + ':'):
            jobs.setdefault(name, []).append((job, comment, state))

    parent = None
    for stage in STAGES:
        if finished(root, stage):
            print(f'{stage}: complete', flush=True)
            continue
        name = f'grpo-{stage}'
        comment = f'{tag}:{parent or "start"}'
        existing = [(job, state) for job, old_comment, state in jobs.get(name, [])
                    if parent is None or old_comment == comment or state != 'PENDING']
        if existing:
            if len(existing) != 1:
                raise RuntimeError(f'Multiple active jobs for {stage}: {existing}')
            parent, state = existing[0]
            print(f'{stage}: reuse {parent} ({state})', flush=True)
            continue
        dep = [f'--dependency=afterok:{parent}'] if parent else []
        result = run('sbatch', '--parsable', '--kill-on-invalid-dep=yes', f'--comment={comment}',
                     f'--job-name={name}', *dep, str(script), stage)
        parent = result.split(';')[0]
        if not parent.isdecimal():
            raise RuntimeError(f'Unexpected sbatch job ID: {result}')
        print(f'{stage}: submitted {parent}', flush=True)


if __name__ == '__main__':
    import fcntl  # login-node lock shared by concurrent launchers
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('script', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / '.submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        submit(args.root.resolve(), args.script.resolve())
