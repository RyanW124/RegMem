"""Build Mem-alpha proposal transitions, then mine session-free repair inputs."""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from tempfile import NamedTemporaryFile

from .proposer import (Consolidator, apply_memory_operations, proposal_messages, repair_context,
                       repair_messages)
from .regression import RegressionGenerator, apply_regression_operations
from .typedefs import (DecisionCase, MemoryItem, MemoryOperation, MemoryState, RegressionSuite)
from .verifier import Verifier


DATASET = 'YuWangX/Memalpha'
REVISION = 'd59b425407f3f1b0e96797e0b817d45062d3f588'
TRAIN_SHA256 = '41d8e9f068018f0463afd5f02cfe2e33953de88a5a6cb7b7ef9daddd13112b0c'
SOURCE_ORDER = ('lme_train', 'perltqa', 'squad', 'hotpotqa')


@dataclass
class Episode:
    id: str
    source: str
    chunks: list[str]


def memory_from_dict(data):
    return MemoryState(items=[MemoryItem(**item) for item in data['items']],
                       version=data['version'], next_id=data['next_id'])


def suite_from_dict(data):
    return RegressionSuite(tests=[DecisionCase(**test) for test in data['tests']],
                           version=data['version'], next_id=data['next_id'])


def proposal_session(row):
    if row['role'] != 'proposal':
        raise ValueError('Expected a proposal row')
    return json.loads(row['prompt'][-1]['content'])['new_evidence']


def read_rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            for item in value if lines else [value]:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_episodes(local_parquet=None):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    # Only request the pinned training file. Its prompt and episode-level answers
    # never enter either policy or the prefix regression suite.
    path = local_parquet or hf_hub_download(DATASET, 'train.parquet', repo_type='dataset', revision=REVISION)
    if sha256(Path(path).read_bytes()).hexdigest() != TRAIN_SHA256:
        raise ValueError('File does not match pinned Mem-alpha train.parquet SHA256')
    records = pq.read_table(path, columns=['instance_id', 'data_source', 'chunks']).to_pylist()
    episodes = []
    for record in records:
        source = record['data_source']
        if source not in SOURCE_ORDER:
            continue
        chunks = json.loads(record['chunks'])
        if not isinstance(chunks, list) or not chunks or not all(isinstance(c, str) and c.strip() for c in chunks):
            raise ValueError(f"Invalid chunks for Mem-alpha instance {record['instance_id']}")
        episodes.append(Episode(f"{source}:{record['instance_id']}", source, chunks))
    if set(SOURCE_ORDER) != {ep.source for ep in episodes}:
        raise ValueError('Missing an expected Mem-alpha training source')
    return episodes


def split_episodes(episodes, seed):
    """Stratify whole episodes; episodes sharing any full chunk must travel together."""
    leaders = {ep.id: ep.id for ep in episodes}
    if len(leaders) != len(episodes):
        raise ValueError('Duplicate episode id')

    def leader(key):
        while key != leaders[key]:
            leaders[key] = leaders[leaders[key]]
            key = leaders[key]
        return key

    seen = {}
    for ep in episodes:
        for chunk in ep.chunks:
            digest = sha256(' '.join(chunk.split()).encode()).digest()
            previous = seen.setdefault(digest, ep.id)
            leaders[leader(ep.id)] = leader(previous)
    groups = defaultdict(list)
    for ep in episodes:
        groups[leader(ep.id)].append(ep)
    counts = Counter(ep.source for ep in episodes)
    target = {source: max(1, round(n / 10)) for source, n in counts.items()}
    selected = Counter()
    rng = random.Random(seed)
    ordered = list(groups.values())
    rng.shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    split = {'train': [], 'dev': []}
    for group in ordered:
        sizes = Counter(ep.source for ep in group)
        # Keep a training group in each source whenever possible.
        fits = all(selected[s] + n < counts[s] for s, n in sizes.items())
        before = sum(abs(target[s] - selected[s]) for s in sizes)
        after = sum(abs(target[s] - selected[s] - n) for s, n in sizes.items())
        if fits and after < before:
            split['dev'].extend(group)
            selected.update(sizes)
        else:
            split['train'].extend(group)
    return split


def _balanced_order(episodes, rng):
    queues = defaultdict(deque)
    episodes = list(episodes)
    rng.shuffle(episodes)
    for ep in episodes:
        queues[ep.source].append(ep)
    while any(queues.values()):
        for source in ('lme_train', 'perltqa', *SOURCE_ORDER):
            if queues[source]:
                yield queues[source].popleft()


def check_verification(suite, result):
    expected = [p.id for p in suite.tests]
    actual = [r.test_id for r in result.results]
    if actual != expected or len(set(actual)) != len(expected):
        raise ValueError('Verifier must return exactly one ordered verdict for each frozen probe')
    return result


def _verified(verifier, memory, suite):
    return check_verification(suite, verifier.verify(memory, suite))


def _prepare_window(ep, start, length, base, generator, verifier):
    """Use a verifiable append-only extraction to build a reachable prefix state."""
    memory, suite = MemoryState(), RegressionSuite()
    for step in range(start, start + length):
        session = ep.chunks[step]
        raw_memory = MemoryState([MemoryItem('0', session)], next_id=1)
        try:
            new_ops = generator.propose(suite, session)[:5]
            extracted = apply_memory_operations(MemoryState(), base.propose(MemoryState(), session))
        except ValueError:
            continue  # malformed extraction/probe generation, independent of policy rewards
        if not new_ops or not extracted.items:
            continue
        tentative = apply_regression_operations(RegressionSuite(), new_ops)
        evidence_pass = _verified(verifier, raw_memory, tentative)
        known_queries = {' '.join(t.query.casefold().split()) for t in suite.tests}
        supported_ops = []
        for op, verdict in zip(new_ops, evidence_pass.results, strict=True):
            query_key = ' '.join(op.query.casefold().split())
            if verdict.passed and query_key not in known_queries:
                known_queries.add(query_key)
                supported_ops.append(op)
        if not supported_ops:
            continue
        writes = [MemoryOperation(action='WRITE', target_id=None, content=i.content) for i in extracted.items]
        witness = apply_memory_operations(memory, writes)
        expanded = apply_regression_operations(suite, supported_ops)
        result = _verified(verifier, witness, expanded)
        passing = {r.test_id for r in result.results if r.passed}
        if any(t.id not in passing for t in suite.tests):
            # A conflicting/unreliable prefix cannot supply a clean preservation target.
            continue
        eligible = [op for op, test in zip(supported_ops, expanded.tests[len(suite.tests):], strict=True)
                    if test.id in passing]
        if not eligible:
            continue
        frozen = apply_regression_operations(suite, eligible)
        yield {'episode_id': ep.id, 'source': ep.source, 'step': step, 'role': 'proposal',
               'prompt': proposal_messages(memory, session), 'memory': asdict(memory),
               'suite': asdict(frozen), 'prior_pass_ids': [t.id for t in suite.tests],
               'target_ids': [t.id for t in frozen.tests[len(suite.tests):]],
               'prepared_memory': asdict(witness)}
        memory, suite = witness, frozen


def prepare(episodes, base, generator, verifier, out, model, seed=42, train_limit=128, dev_limit=32, max_chunks=4):
    split = split_episodes(episodes, seed)
    manifest = {'dataset': DATASET, 'revision': REVISION, 'source_split': 'train', 'model': model,
                'seed': seed, 'max_chunks': max_chunks,
                'episodes': {name: sorted(ep.id for ep in group) for name, group in split.items()}, 'rows': {}}
    for name, limit in [('train', train_limit), ('dev', dev_limit)]:
        rng = random.Random(seed + int(name == 'dev'))
        rows = []
        for ep in _balanced_order(split[name], rng):
            length = min(max_chunks, len(ep.chunks))
            start = rng.randrange(len(ep.chunks) - length + 1)
            for row in _prepare_window(ep, start, length, base, generator, verifier):
                rows.append(row)
                if len(rows) == limit:
                    break
            if len(rows) == limit:
                break
        if not rows:
            raise ValueError(f'No verified {name} transitions from eligible episodes')
        _write_json(out / f'proposal_{name}.jsonl', rows, lines=True)
        manifest['rows'][name] = len(rows)
    _write_json(out / 'episode_splits.json', manifest)
    return manifest


def _restoration_patches(original, candidate):
    """Enumerate short, executable witness patches using only available item text."""
    current = {i.id: i.content for i in candidate.items}
    original_ids = {i.id for i in original.items}
    restore = []
    for item in original.items:
        if current.get(item.id) == item.content:
            continue
        if item.id in current:
            restore.append(MemoryOperation(action='REVISE', target_id=item.id, content=item.content))
        else:
            restore.append(MemoryOperation(action='WRITE', target_id=None, content=item.content))
    if 1 <= len(restore) <= 10:
        yield restore
    if len(restore) > 1:
        for op in restore:
            yield [op]
    # A false new item can mask a correct answer still in the candidate.
    for item in candidate.items:
        if item.id not in original_ids:
            yield [MemoryOperation(action='PRUNE', target_id=item.id, content=None)]


def recoverable_failure(original, candidate, suite, verifier):
    """Return whether an edit of just repair-visible items can verifiably pass."""
    for patch in _restoration_patches(original, candidate):
        restored = apply_memory_operations(candidate, patch)
        if _verified(verifier, restored, suite).passed:
            return True
    return False


def classify_failure(row, candidate, verification, verifier, origin='mined'):
    """Label demonstrably recoverable failures. A missing witness is a conservative negative."""
    original, suite = memory_from_dict(row['memory']), suite_from_dict(row['suite'])
    check_verification(suite, verification)
    if verification.passed:
        raise ValueError('Expected a failed candidate')
    failures, previous = repair_context(original, candidate, suite, verification)
    witnessed = recoverable_failure(original, candidate, suite, verifier)
    # Explicit construction excludes proposal prompts, raw input, teacher states,
    # reference answers in the prompt, and verifier reasons from repair feedback.
    return {'episode_id': row['episode_id'], 'source': row.get('source'), 'step': row['step'], 'role': 'repair',
            'prompt': repair_messages(candidate, failures, previous), 'memory': asdict(candidate),
            'suite': asdict(suite), 'target_ids': verification.failed_test_ids,
            'prior_pass_ids': [r.test_id for r in verification.results if r.passed],
            'recoverable': witnessed, 'origin': origin}


def _synthetic(row, verifier, rng):
    original = memory_from_dict(row['memory'])
    witness = memory_from_dict(row['prepared_memory'])
    suite = suite_from_dict(row['suite'])
    previous = {i.id: i.content for i in original.items}
    stable = [i for i in witness.items if previous.get(i.id) == i.content]
    rng.shuffle(stable)
    old_ids = set(row['prior_pass_ids'])
    labels = [t.expected_answer.casefold() for t in suite.tests if t.id in old_ids]
    stable.sort(key=lambda item: any(label in item.content.casefold() for label in labels), reverse=True)
    for item in stable:
        # Prefer the smallest realistic edit when an item has an explicit qualifier.
        parts = re.split(r'\s+(?:except|unless|only when|provided that)\b', item.content,
                         maxsplit=1, flags=re.IGNORECASE)
        action, content, origin = 'PRUNE', None, 'synthetic_deletion'
        if len(parts) == 2 and parts[0].strip(' ,;.'):
            action, content, origin = 'REVISE', parts[0].rstrip(' ,;.') + '.', 'synthetic_qualifier'
        op = MemoryOperation(action=action, target_id=item.id, content=content)
        candidate = apply_memory_operations(witness, [op])
        verification = _verified(verifier, candidate, suite)
        if not verification.passed:
            sample = classify_failure(row, candidate, verification, verifier, origin=origin)
            if sample['recoverable']:
                return sample
    return None


def mine_split(rows, base, verifier, limit, seed):
    failures = []
    for row in rows:
        memory = memory_from_dict(row['memory'])
        suite = suite_from_dict(row['suite'])
        try:
            operations = base.propose(memory, proposal_session(row))
        except ValueError:
            operations = []  # same invalid-proposal behavior as the live engine
        candidate = apply_memory_operations(memory, operations)
        verdict = _verified(verifier, candidate, suite)
        if not verdict.passed:
            failures.append(classify_failure(row, candidate, verdict, verifier))
    repair_rows = [r for r in failures if r['recoverable']]
    rng = random.Random(seed)
    rng.shuffle(repair_rows)
    repair_rows = repair_rows[:limit]
    candidates = list(rows)
    rng.shuffle(candidates)
    seen = {(r['episode_id'], r['step'], r['prompt'][-1]['content']) for r in repair_rows}
    for proposal in candidates:
        if len(repair_rows) >= limit:
            break
        synthetic = _synthetic(proposal, verifier, rng)
        if synthetic is not None:
            key = (synthetic['episode_id'], synthetic['step'], synthetic['prompt'][-1]['content'])
            if key not in seen:
                seen.add(key)
                repair_rows.append(synthetic)
    return repair_rows, failures


def mine(out, base, verifier, model, adapter, seed=42, train_limit=128, dev_limit=32):
    manifest = json.loads((out / 'episode_splits.json').read_text(encoding='utf-8'))
    if manifest['revision'] != REVISION or manifest['model'] != model:
        raise ValueError('Preparation manifest must match the dataset revision and model used for mining')
    report = {'proposal_adapter': str(adapter), 'model': model, 'seed': seed, 'splits': {}}
    for name, limit in [('train', train_limit), ('dev', dev_limit)]:
        rows = read_rows(out / f'proposal_{name}.jsonl')
        allowed = set(manifest['episodes'][name])
        if any(r['episode_id'] not in allowed for r in rows):
            raise ValueError(f'Proposal rows disagree with {name} episode manifest')
        repairs, failures = mine_split(rows, base, verifier, limit, seed)
        if name == 'dev':
            _write_json(out / 'repair_failures_dev.jsonl', failures, lines=True)
        _write_json(out / f'repair_{name}.jsonl', repairs, lines=True)
        report['splits'][name] = {'proposals': len(rows), 'failures': len(failures),
                                  'witnessed_failures': sum(r['recoverable'] for r in failures),
                                  'repair_rows': dict(Counter(r['origin'] for r in repairs))}
    _write_json(out / 'repair_mining.json', report)
    if not all(s['repair_rows'] for s in report['splits'].values()):
        raise ValueError(f'No recoverable rows in at least one split; see {out / "repair_mining.json"}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'mine-repair'))
    parser.add_argument('--data-dir', type=Path, default=Path('runs/grpo/data'))
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--proposal-adapter', type=Path)
    parser.add_argument('--local-parquet', type=Path, help='Local copy of the pinned Mem-alpha train parquet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-limit', type=int, choices=range(1, 129), metavar='1..128', default=128)
    parser.add_argument('--dev-limit', type=int, choices=range(1, 33), metavar='1..32', default=32)
    parser.add_argument('--max-chunks', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.stage == 'mine-repair' and args.proposal_adapter is None:
        parser.error('mine-repair requires --proposal-adapter')
    from transformers import set_seed
    from .llm import LLM
    set_seed(args.seed)
    llm = LLM(args.model, temperature=0)
    verifier = Verifier(llm)
    if args.stage == 'prepare':
        report = prepare(load_episodes(args.local_parquet), Consolidator(llm), RegressionGenerator(llm), verifier, args.data_dir,
                         args.model, args.seed, args.train_limit, args.dev_limit, args.max_chunks)
        print(report['rows'])
    else:
        policy = Consolidator(llm.for_adapter(args.proposal_adapter, 'proposal'), llm)
        report = mine(args.data_dir, policy, verifier, args.model, args.proposal_adapter,
                      args.seed, args.train_limit, args.dev_limit)
        print(report['splits'])


if __name__ == '__main__':
    main()
