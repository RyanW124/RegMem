"""MemConflict benchmark: run arms on one instance at one or more memory token budgets.

  probes     generate the shared probe suite once (first k per session that the session's own extraction
             passes) and cache that extraction -> {out}/probes/inst{i}.json
  extract    arm (a), append-only: replays each session's cached extraction (empty memory, so WRITE-only)
  summarize  arm (b), unverified consolidation: Consolidator output applied every session
  regmem     arm (c): cumulative regression suite gate; a rejected update rolls back
  trustmem   arm (d): TrustMem-style check of each transition (old memory, session, candidate); a rejected
             update rolls back

Each arm runs once per --budget (0 = no cap). Under a budget the consolidator is told it, a candidate over
it fails the check (so repair has to compress), and as a last resort any memory still over it drops its
oldest items, which is append-only's whole policy. After each session every arm logs memory and answers
that session's MemConflict queries (the benchmark protocol). At checkpoints it measures pass rate on the
shared probe suite and re-asks the static queries issued so far. Probes come first (same --out); several
arms can share one process, e.g. --arm probes extract summarize regmem trustmem --budget 1000 2000.
"""
import argparse
import hashlib
import json
import random
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from regmem import ConsolidationEngine, Consolidator, RegressionGenerator, SuiteCurator, TransitionChecker, Verifier
from regmem.proposer import apply_memory_operations
from regmem.regression import apply_regression_operations
from regmem.typedefs import (ConsolidationOutput, JudgeResult, MemoryOperation, MemoryState, RegressionSuite,
                             RegressionSuiteOperation, RegressionSuiteOutput, SuiteEdit, SuiteEditOutput,
                             TransitionVerdict)

# Answer and judge prompts from MemConflict (github.com/TaoZhen1110/MemConflict: Evaluation/eval_memzero.py,
# Evaluation/eval_scoring.py), with the white-box retrieval inputs and metrics removed.
ANSWER_SYSTEM_PROMPT = """You answer memory-evaluation questions using only the retrieved memory context.

Rules:
1. Use only the retrieved memories.
2. Do not invent facts that are not supported by the retrieved memories.
3. If the memories are insufficient, say that you cannot confirm.
4. If the memories contain inconsistent statements, briefly mention the inconsistency first and then give the best-supported answer.
5. Keep the answer concise, natural, and directly responsive to the question."""

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for memory-conflict experiments.
You must score only the requested metrics.
Use the reference answer as the gold standard.
Return valid JSON only."""

JUDGE_METRICS = {
    'dynamic_conflict': '''- dynamic_answer_accuracy: Score 1.0 if the model answer captures the core updated fact correctly according to the reference answer. Score 0.5 if it is partially correct: it mentions a correct key entity, destination, new state, or update direction, but misses important old/new details or is too incomplete for full credit. Score 0.0 if it is wrong, contradictory, does not contain the key updated fact, or is overly uncertain. Do not require exact wording.
- update_awareness_and_order_consistency_score: 1 if the model answer clearly shows that an update/change happened and preserves the correct old-to-new direction. Do not require every old/new detail to be repeated, but the answer must still indicate the update in the right direction. Otherwise 0.''',
    'static_conflict': '''- static_answer_accuracy: Score 1.0 if the model answer captures the core true fact or judgment correctly according to the reference answer. Score 0.5 if it contains a correct key fact or partially correct semantic match, but is incomplete, vague, or misses an important detail needed for full correctness. Score 0.0 if it is wrong, contradictory, does not contain the key true fact, or is overly uncertain. Do not require exact wording.
- conflict_recognition_score: 1 if the model answer appropriately recognizes the inconsistency or uncertainty that exists in this static-conflict case; otherwise 0.''',
    'conditional_conflict': '''- conditional_answer_accuracy: Score 1.0 if the model answer gives the correct condition according to the reference answer in a semantically correct way. If the answer captures any correct core condition that satisfies the reference, give 1.0. Score 0.0 if the condition is wrong, contradictory, absent, or overly uncertain. Do not require exact wording.''',
}


# MemConflict lists reasoning last; it comes first here so a local model's constrained JSON reasons
# before it scores (the metric definitions are unchanged).
class DynamicJudge(BaseModel):
    reasoning: str = Field(max_length=400)
    dynamic_answer_accuracy: Literal[0.0, 0.5, 1.0]
    update_awareness_and_order_consistency_score: Literal[0, 1]


class StaticJudge(BaseModel):
    reasoning: str = Field(max_length=400)
    static_answer_accuracy: Literal[0.0, 0.5, 1.0]
    conflict_recognition_score: Literal[0, 1]


class ConditionalJudge(BaseModel):
    reasoning: str = Field(max_length=400)
    conditional_answer_accuracy: Literal[0.0, 1.0]


JUDGES = {'dynamic_conflict': DynamicJudge, 'static_conflict': StaticJudge, 'conditional_conflict': ConditionalJudge}


def judge_prompt(question, answer):
    kind = question['conflict_type']
    schema = {name: ('short explanation' if name == 'reasoning' else 0.0 if 'accuracy' in name else 0)
              for name in JUDGES[kind].model_fields}
    return (f"You are evaluating one {kind.replace('_', '-')} question.\n\nInputs:\n"
            f"1. Question: {question['question']}\n2. Reference Answer: {question['answer']}\n"
            f"3. Model Answer: {answer}\n\nMetric definitions:\n{JUDGE_METRICS[kind]}\n\n"
            f"Return JSON exactly in this schema:\n{json.dumps(schema, indent=2)}")


def answer_prompt(memory, question):
    context = ('Memories:\n' + '\n'.join(f'{n}. {x.content}' for n, x in enumerate(memory.items, 1))
               if memory.items else 'No relevant memories found.')
    return f"Retrieved Memory Context:\n{context}\n\nQuestion:\n{question['question']}\n\nAnswer:"


def ask(llm, memory, questions):
    """Answer MemConflict questions from memory and score each with its type-specific judge."""
    answers = llm.generate_batch(ANSWER_SYSTEM_PROMPT, [answer_prompt(memory, q) for q in questions])
    rows = [{'question_id': q['question_id'], 'conflict_type': q['conflict_type'], 'question': q['question'],
             'reference': q['answer'], 'answer': a} for q, a in zip(questions, answers)]
    for kind, schema in JUDGES.items():
        picked = [n for n, q in enumerate(questions) if q['conflict_type'] == kind]
        if not picked:  # also avoids compiling a judge's output grammar before it is needed
            continue
        verdicts = llm.generate_structured_batch(
            JUDGE_SYSTEM_PROMPT, [judge_prompt(questions[n], answers[n]) for n in picked], schema)
        for n, verdict in zip(picked, verdicts):
            rows[n].update(verdict.model_dump(), accuracy=getattr(verdict, kind.split('_')[0] + '_answer_accuracy'))
    return rows


def session_evidence(session):
    turns = session.get('Session_Dialogue') or {}
    lines = [f"{m['role']}: {m['content']}" for key in sorted(turns, key=lambda k: int(k.split('_')[-1]))
             for m in turns[key] if m.get('role') in ('user', 'assistant') and m.get('content')]
    return f"[Session date: {session.get('Date')}]\n" + '\n'.join(lines)


class CachedProbes:
    """Stands in for RegressionGenerator so every arm is measured on the identical probe suite."""
    def __init__(self, ops_by_evidence):
        self.ops_by_evidence = ops_by_evidence

    def propose(self, suite, evidence):
        return self.ops_by_evidence.get(evidence, [])  # arms without tests get none


def extract(consolidator, memory, evidence):
    """Arm (a): extract this session alone (empty memory, so WRITE-only) and append the items."""
    return apply_memory_operations(memory, consolidator.propose(MemoryState(), evidence))


class StubLLM:
    """Deterministic fake LLM for a local dry run of the whole pipeline (no GPU, no model)."""
    @staticmethod
    def _roll(text):
        return int(hashlib.md5(text.encode()).hexdigest(), 16) % 100

    def generate(self, system_prompt, user_prompt):
        return self.generate_batch(system_prompt, [user_prompt])[0]

    def generate_structured(self, system_prompt, user_prompt, schema):
        return self.generate_structured_batch(system_prompt, [user_prompt], schema)[0]

    def generate_batch(self, system_prompt, user_prompts):
        return [f'stub answer {self._roll(p)}' for p in user_prompts]

    def generate_structured_batch(self, system_prompt, user_prompts, schema):
        return [self._fake(p, schema) for p in user_prompts]

    def _fake(self, prompt, schema):
        roll = self._roll(prompt)
        if schema is RegressionSuiteOutput:
            return schema(operations=[RegressionSuiteOperation(action='WRITE', target_id=None, query=f'stub probe {roll}.{n}?',
                                                               expected_answer=f'stub {roll}.{n}') for n in range(4)])
        if schema is ConsolidationOutput:
            payload = json.loads(prompt)
            if any({'expected_answer', 'reason'} & set(f) for f in payload.get('failed_checks', [])):
                raise SystemExit('gold answer leaked into the repair payload')  # not caught by the runner
            ops = [MemoryOperation(action='WRITE', target_id=None, content=f'stub fact {roll} ' + 'word ' * (roll % 20))]
            # Prune the oldest items when over budget, so the budget-repair path converges.
            n_prune = 3 if payload.get('over_budget_by_tokens', 0) > 0 else int(bool(payload['memory']) and roll < 30)
            ops += [MemoryOperation(action='PRUNE', target_id=x['id'], content=None) for x in payload['memory'][:n_prune]]
            if roll >= 97:  # an invalid op now and then, to exercise error handling
                ops.append(MemoryOperation(action='REVISE', target_id='no-such-id', content='x'))
            return schema(operations=ops)
        if schema is SuiteEditOutput:
            payload = json.loads(prompt)
            old = [t for t in payload['tests'] if not t['new']]
            edits = [SuiteEdit(reason='stub change', target_id=old[0]['id'], action='REVISE', expected_answer=f'stub new {roll}')] if old and roll < 20 else []
            extra = len(payload['tests']) - (payload['max_tests'] or len(payload['tests']))
            edits += [SuiteEdit(reason='stub minor', target_id=t['id'], action='PRUNE', expected_answer=None)
                      for t in payload['tests'][1:1 + max(0, extra - 1)]]  # one short, to exercise the fallback
            if roll >= 95:
                edits.append(SuiteEdit(reason='stub bad id', target_id='no-such-id', action='PRUNE', expected_answer=None))
            return schema(edits=edits)
        if schema is TransitionVerdict:
            return schema(problems=['stub: a new fact is missing'] if roll < 30 else [])
        if schema is JudgeResult:
            return schema(passed=roll < 85, reason='stub verdict')
        return schema(**{name: 'stub' if name == 'reasoning' else (1.0 if roll < 50 else 0.0) if 'accuracy' in name
                         else int(roll < 50) for name in schema.model_fields})


def load_llm(args):
    random.seed(args.seed)
    if args.model == 'stub':
        return StubLLM()
    from transformers import set_seed
    from regmem.llm import LLM
    set_seed(args.seed)
    return LLM(args.model, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)


def memory_tokens(llm, memory):
    text = '\n'.join(x.content for x in memory.items)
    tokenizer = getattr(llm, 'tokenizer', None)
    return len(tokenizer(text).input_ids) if tokenizer is not None else len(text) // 4


def trim(memory, budget, count):
    """Last resort for every arm, and append-only's whole policy: drop the oldest items until memory fits."""
    if not budget or count(memory) <= budget:
        return memory, 0
    memory = deepcopy(memory)
    dropped = 0
    while memory.items and count(memory) > budget:
        memory.items.pop(0)
        dropped += 1
    return memory, dropped


def make_probes(args, llm, instance, sessions, path):
    generator, suite, per_session, extractions, errors = RegressionGenerator(llm), RegressionSuite(), [], [], []
    consolidator, verifier = Consolidator(llm), Verifier(llm)
    for c, session in enumerate(sessions):
        evidence, ops, facts = session_evidence(session), [], []
        try:
            # Arm (a)'s extraction of this session, cached so every arm and budget replays the same facts.
            facts = consolidator.propose(MemoryState(), evidence)
            ops = generator.propose(suite, evidence)
            if ops:
                # Keep only probes this session's own extracted facts answer: a regression is a test that
                # once passed, and a probe no memory passes would block every RegMem cycle.
                memory = apply_memory_operations(MemoryState(), facts)
                failed = set(verifier.verify(memory, apply_regression_operations(RegressionSuite(), ops)).failed_test_ids)
                ops = [op for n, op in enumerate(ops) if str(n) not in failed]
            ops = ops[:args.probes_per_session]
        except Exception as error:  # a malformed generation costs this session's probes, not the run
            ops = []
            errors.append({'cycle': c, 'error': f'{type(error).__name__}: {error}'})
        suite = apply_regression_operations(suite, ops)
        per_session.append([op.model_dump() for op in ops])
        extractions.append([op.model_dump() for op in facts])
        print(f'[probes inst{args.instance}] session {c + 1}/{len(sessions)}: {len(ops)} probes, '
              f'{len(facts)} facts', flush=True)
    path.write_text(json.dumps({'args': vars(args), 'instance_id': instance.get('ID'), 'errors': errors,
                                'sessions': per_session, 'extractions': extractions}, ensure_ascii=False, indent=1))
    print(f'wrote {path} ({len(suite.tests)} probes, {len(errors)} errored sessions)')


def run_arm(args, arm, budget, llm, instance, sessions, probes, out):
    name = f'{arm}_b{budget}' if budget else arm  # uncapped runs keep the midterm names
    trace_dir = out / 'traces' / f'inst{args.instance}_{name}'
    ops_by_cycle = [[RegressionSuiteOperation(**op) for op in ops] for ops in probes['sessions'][:len(sessions)]]
    cached = probes.get('extractions')  # probes files from the midterm have none: extract live instead
    consolidator, verifier = Consolidator(llm), Verifier(llm)

    def count(memory):
        return memory_tokens(llm, memory)

    def append_session(memory, c, evidence):
        if cached is None:
            return extract(consolidator, memory, evidence)
        return apply_memory_operations(memory, [MemoryOperation(**op) for op in cached[c]])

    tests = CachedProbes({session_evidence(s): ops for s, ops in zip(sessions, ops_by_cycle)} if arm == 'regmem' else {})
    engine = ConsolidationEngine(consolidator, tests, verifier, max_repairs=args.max_repairs,
                                 curator=SuiteCurator(llm) if arm == 'regmem' else None,
                                 transition_checker=TransitionChecker(llm) if arm == 'trustmem' else None,
                                 token_budget=budget or None, count_tokens=count,
                                 test_budget=budget // args.tokens_per_test if budget and arm == 'regmem' else None)
    memory, gate, shared, issued_static = MemoryState(), RegressionSuite(), RegressionSuite(), []
    path = out / 'arms' / f'inst{args.instance}_{name}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as log:
        log.write(json.dumps({'header': True, 'arm': arm, 'budget': budget, 'test_budget': engine.test_budget,
                              'args': vars(args), 'instance_id': instance.get('ID'), 'n_sessions': len(sessions)}) + '\n')
        for c, session in enumerate(sessions):
            evidence, ops, start = session_evidence(session), ops_by_cycle[c], time.time()
            shared = apply_regression_operations(shared, ops)
            row = {'cycle': c, 'session_id': session.get('Session_ID'), 'date': session.get('Date')}
            try:
                if arm == 'extract':
                    memory, row['status'] = append_session(memory, c, evidence), 'appended'
                else:
                    result = engine.consolidate(memory, gate, evidence, save_run=True, output_dir=trace_dir)
                    row.update(attempts=result.attempts, trace=result.run_path,
                               failed_test_ids=result.verification.failed_test_ids, **result.blocking)
                    gate = result.suite
                    if result.accepted:
                        memory, row['status'] = result.memory, 'accepted'
                    elif arm == 'summarize':  # nothing checks content, so only the budget failed: trim below
                        memory, row['status'] = apply_memory_operations(memory, result.operations), 'over_budget'
                    else:  # roll back: keep the previous memory
                        row['status'] = 'rejected'
            except Exception as error:  # malformed or invalid LLM output must not end the run
                row.update(status='error', error=f'{type(error).__name__}: {error}')
                if arm == 'summarize':  # keep the session's facts as arm (a) would; checked arms roll back
                    try:
                        memory = append_session(memory, c, evidence)
                    except Exception as fallback_error:
                        row['fallback_error'] = f'{type(fallback_error).__name__}: {fallback_error}'
            gate.next_id = shared.next_id  # keep gate test ids equal to shared-suite ids, even after errors
            memory, row['trimmed_items'] = trim(memory, budget, count)
            checkpoint = (c + 1) % args.checkpoint_every == 0 or c == len(sessions) - 1
            questions = session.get('Session_Questions') or []
            issued_static += [q for q in questions if q.get('conflict_type') == 'static_conflict']
            try:
                row['queries'] = ask(llm, memory, questions)
                if checkpoint and shared.tests:
                    verification = verifier.verify(memory, shared)
                    row['probes'] = [asdict(r) for r in verification.results]
                    row['probe_pass_rate'] = sum(r.passed for r in verification.results) / len(verification.results)
                if checkpoint:
                    row['static_reask'] = ask(llm, memory, issued_static)
            except Exception as error:
                row['eval_error'] = f'{type(error).__name__}: {error}'
            row.update(checkpoint=checkpoint, memory=[asdict(x) for x in memory.items],
                       memory_tokens=memory_tokens(llm, memory), gate_size=len(gate.tests),
                       shared_size=len(shared.tests), seconds=round(time.time() - start, 1))
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            print(f"[{name} inst{args.instance}] cycle {c + 1}/{len(sessions)} {row['status']}"
                  f" | memory {len(memory.items)} items/{row['memory_tokens']} tok | gate {len(gate.tests)}"
                  + (f" | probe pass {row['probe_pass_rate']:.2f}" if 'probe_pass_rate' in row else '')
                  + (f" | {row.get('error') or row.get('eval_error')}" if 'error' in row or 'eval_error' in row else '')
                  + f" | {row['seconds']}s", flush=True)
    print(f'wrote {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', required=True, nargs='+', choices=['probes', 'extract', 'summarize', 'regmem', 'trustmem'],
                        help='one or more, run in one process (startup costs minutes on the cluster)')
    parser.add_argument('--budget', type=int, nargs='+', default=[0],
                        help='memory token budgets; each arm runs once per budget (0 = no cap)')
    parser.add_argument('--tokens-per-test', type=int, default=50,
                        help='under a budget, RegMem keeps at most budget / this many tests')
    parser.add_argument('--instance', type=int, required=True, help='line index in the MemConflict JSONL')
    parser.add_argument('--data', default='data/memconflict/Step4_4.jsonl')
    parser.add_argument('--model', default='Qwen/Qwen3-8B', help="HF model id, or 'stub' for a local dry run")
    parser.add_argument('--out', default='results/memconflict', help='run directory shared by all arms')
    parser.add_argument('--max-sessions', type=int, help='only the first N sessions (smoke/pilot)')
    parser.add_argument('--probes-per-session', type=int, default=3)
    parser.add_argument('--max-repairs', type=int, default=2)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.data, encoding='utf-8') as stream:
        line = next((line for n, line in enumerate(stream) if n == args.instance), None)
    if line is None:
        raise SystemExit(f'--instance {args.instance} is out of range for {args.data}')
    instance = json.loads(line)
    sessions = instance['Full_Session_Chain'][:args.max_sessions]
    out = Path(args.out)
    probes_path = out / 'probes' / f'inst{args.instance}.json'
    arms = [arm for arm in args.arm if arm != 'probes']
    if 'probes' not in args.arm and not probes_path.exists():  # fail before the slow model load
        raise SystemExit(f'{probes_path} missing: run --arm probes first with the same --out')
    llm = load_llm(args)
    if 'probes' in args.arm:
        probes_path.parent.mkdir(parents=True, exist_ok=True)
        make_probes(args, llm, instance, sessions, probes_path)
    if arms:
        probes = json.loads(probes_path.read_text())
        if len(probes['sessions']) < len(sessions):
            raise SystemExit(f'{probes_path} covers {len(probes["sessions"])} sessions; rerun --arm probes')
        if probes['args']['model'] != args.model:
            print(f"warning: probes were generated with {probes['args']['model']}, these arms run {args.model}")
        for budget in args.budget:
            for arm in arms:
                run_arm(args, arm, budget, llm, instance, sessions, probes, out)


if __name__ == '__main__':
    main()
