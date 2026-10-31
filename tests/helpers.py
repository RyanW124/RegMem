"""Shared doubles and builders for the regmem tests."""
import json
from collections import deque

from regmem.typedefs import (ConsolidationOutput, DecisionCase, JudgeResult, MemoryItem, MemoryOperation,
                            MemoryState, RegressionSuite, RegressionSuiteOperation, RegressionSuiteOutput)


def memory(*contents, version=0, next_id=None):
    """MemoryState whose items have ids '0', '1', ... in the order given."""
    items = [MemoryItem(str(n), content) for n, content in enumerate(contents)]
    return MemoryState(items=items, version=version, next_id=len(items) if next_id is None else next_id)


def suite(*pairs, version=0, next_id=None):
    """RegressionSuite of (query, expected_answer) probes with ids '0', '1', ..."""
    tests = [DecisionCase(str(n), query, answer) for n, (query, answer) in enumerate(pairs)]
    return RegressionSuite(tests=tests, version=version, next_id=len(tests) if next_id is None else next_id)


def write(content, target_id=None):
    return MemoryOperation(action='WRITE', content=content, target_id=target_id)


def revise(target_id, content):
    return MemoryOperation(action='REVISE', content=content, target_id=target_id)


def prune(target_id, content=None):
    return MemoryOperation(action='PRUNE', content=content, target_id=target_id)


def probe(query, expected_answer, action='WRITE', target_id=None):
    return RegressionSuiteOperation(action=action, target_id=target_id, query=query, expected_answer=expected_answer)


class ScriptedLLM:
    """Language-model double.

    Proposals come from queues passed in; the queue is consumed one item per call and an Exception
    instance in the queue is raised instead of returned. Answers come from `answer(query, memory)`
    and verdicts from `verdict(query, expected, predicted)`; `solver_batch`/`judge_batch` replace the
    whole batched response so tests can control its length. Every prompt is recorded per role.
    """

    def __init__(self, memory_proposals=(), repairs=(), test_proposals=(), answer=None, verdict=None,
                 solver_batch=None, judge_batch=None):
        self.memory_proposals = deque(memory_proposals)
        self.repairs = deque(repairs)
        self.test_proposals = deque(test_proposals)
        self.answer = answer or (lambda query, items: 'unknown')
        self.verdict = verdict or (lambda query, expected, predicted: predicted == expected)
        self.solver_batch = solver_batch
        self.judge_batch = judge_batch
        self.prompts = {'solver': [], 'judge': [], 'propose': [], 'repair': [], 'tests': []}

    # -- LanguageModel protocol -------------------------------------------------------------
    def generate(self, system_prompt, user_prompt):
        return self.generate_batch(system_prompt, [user_prompt])[0]

    def generate_structured(self, system_prompt, user_prompt, schema):
        if schema is RegressionSuiteOutput:
            self.prompts['tests'].append(user_prompt)
            return RegressionSuiteOutput(operations=self._next(self.test_proposals))
        if schema is ConsolidationOutput:
            payload = json.loads(user_prompt)
            role = 'repair' if 'task' in payload else 'propose'
            self.prompts[role].append(user_prompt)
            return ConsolidationOutput(operations=self._next(self.repairs if role == 'repair'
                                                            else self.memory_proposals))
        raise AssertionError(f'unexpected schema {schema}')

    def generate_batch(self, system_prompt, user_prompts):
        self.prompts['solver'].extend(user_prompts)
        if self.solver_batch is not None:
            return self.solver_batch(list(user_prompts))
        answers = []
        for prompt in user_prompts:
            payload = json.loads(prompt)
            answers.append(self.answer(payload['query'], payload['memory']))
        return answers

    def generate_structured_batch(self, system_prompt, user_prompts, schema):
        assert schema is JudgeResult, schema
        self.prompts['judge'].extend(user_prompts)
        if self.judge_batch is not None:
            return self.judge_batch(list(user_prompts))
        verdicts = []
        for prompt in user_prompts:
            payload = json.loads(prompt)
            passed = self.verdict(payload['query'], payload['expected_answer'], payload['predicted_answer'])
            verdicts.append(JudgeResult(passed=passed, reason='scripted verdict'))
        return verdicts

    @staticmethod
    def _next(queue):
        if not queue:
            return []
        item = queue.popleft()
        if isinstance(item, BaseException):
            raise item
        return item


def answers_from(mapping):
    """answer() double: returns mapping[query], or 'unknown' for a query it was not told about."""
    return lambda query, items: mapping.get(query, 'unknown')
