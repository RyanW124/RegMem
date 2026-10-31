# RegMem

Regression-guided memory consolidation that preserves decision-relevant information.

The regression suite is cumulative and append-only. Each evidence batch generates
new probes, then an LLM proposes WRITE / REVISE / PRUNE memory operations. A solver
answers every old and new probe using only candidate memory. An LLM judge scores
semantic correctness, including constraints and exceptions. Failed candidates get
up to `max_repairs` repair patches and are checked against the entire frozen suite
again. Acceptance requires every test to pass.

```python
from regmem import ConsolidationEngine, Consolidator, RegressionGenerator, Verifier
from regmem.llm import LLM
from regmem.typedefs import MemoryState, RegressionSuite

llm = LLM('Qwen/Qwen3-0.6B')
engine = ConsolidationEngine(
    Consolidator(llm), RegressionGenerator(llm), Verifier(llm), max_repairs=2)
memory, suite = MemoryState(), RegressionSuite()

for evidence in ['I prefer tea.', 'I cannot have milk.']:
    result = engine.consolidate(memory, suite, evidence)
    if result.accepted:
        memory, suite = result.memory, result.suite
    else:
        print('Rejected:', result.verification.failed_test_ids)
```

Inputs are never modified. A rejection returns copies of the original memory and
suite; callers should retain rejected evidence for retry. Model/schema errors
raise without committing either state. Repair operations apply sequentially to
the candidate; the returned operation list can be replayed against original memory.
An accepted cycle increments memory version once when operations were proposed.

`Verifier(llm, judge=another_llm)` supports a separate judge. The solver never sees
reference answers or raw evidence. The judge sees query, reference, and prediction.
Generated tests and LLM judgments are fallible: passing the suite establishes only
preservation covered by those probes. An empty suite passes vacuously.

Install the model dependencies from `requirements.txt` in your GPU environment.
The existing LLM wrapper uses Transformers and Outlines; the core only needs
Pydantic 2. 

To save the details of a consolidation run:

```python
result = engine.consolidate(
    memory, suite, evidence, save_run=True, output_dir='results/runs')
print(result.run_path)
```

`save_run` defaults to `False` (no files are written and `run_path` is `None`).
Each enabled call writes a uniquely named JSON file containing the evidence,
initial memory and suite, generated test operations, initial memory operations,
candidate snapshots, every completed verification (answers, pass/fail and judge
reasons), repair patches and their failure feedback, and final accepted/rejected
state. Timestamps are UTC. Relative output paths resolve from the working directory.
Errored runs save the completed events plus exception type/message before reraising;
operations or partial verifier results that never returned cannot be recorded.
File-writing errors propagate. Logs contain the supplied memory and evidence.

## Install

```bash
pip install .                 # core: Pydantic 2 only
pip install '.[models]'       # + Transformers, Outlines, torch for a real model
pip install '.[dev]'          # + pytest and ruff
```

This also installs the `regmem` command. `run_memconflict.py` and
`analyze_memconflict.py` stay plain scripts, so the benchmark keeps working from a
checkout with `requirements.txt`.

## Command line

```bash
regmem consolidate --memory memory.json --suite suite.json --evidence note.txt --save-run
regmem consolidate --memory memory.json --suite suite.json --text 'I prefer tea.' --write --json
regmem verify --memory memory.json --suite suite.json
regmem inspect results/runs/run-*.json
```

`consolidate` runs one cycle and reports the outcome, the memory and suite size
change, and the probe result. It leaves the files alone unless `--write` is given,
which saves the accepted memory and suite back (a rejected cycle writes nothing).
`verify` re-runs the suite against a saved memory without proposing anything.
`inspect` summarizes saved traces: status, attempts, probe counts, and the ids that
failed, or the recorded error for an errored run. Exit codes are `0` for accepted
(all probes passed), `1` for rejected (a probe failed), and `2` for a usage, file or
model error, so a script can gate on the outcome without parsing output.

## Saving and loading memory

Memory and suites are plain dataclasses, so `regmem.persistence` writes them as JSON
and reads them back:

```python
from regmem import load_memory, load_suite, save_memory, save_suite

memory, suite = load_memory('memory.json'), load_suite('suite.json')
result = engine.consolidate(memory, suite, evidence)
if result.accepted:
    save_memory('memory.json', result.memory)
    save_suite('suite.json', result.suite)
```

Writes go through a temporary file and `os.replace` in the destination directory, so
an interrupted or concurrent write cannot leave a half-written memory behind, and
concurrent traces never overwrite each other. Loading validates the whole payload:
field types, non-blank text, unique ids, and `next_id` above every integer id (new
items are numbered from it, so a smaller value would collide on the next WRITE).
A malformed file raises `ValueError` naming the field.

## Dry runs without a model

`regmem.stub.StubLLM` answers, proposes and judges as a pure function of the prompt,
so a full cycle can run on a laptop with no GPU, no weights and no network:

```bash
regmem consolidate --memory memory.json --suite suite.json --text 'I prefer tea.' --model stub
python run_memconflict.py --arm probes extract summarize regmem --instance 0 --model stub
```

It exercises the plumbing (proposal validation, the repair loop, traces, batching),
not the quality of the memory it builds. It also emits an invalid operation now and
then so the invalid-proposal path stays covered.

## What a pass means

The judge is asked for one verdict per probe, and verification requires exactly one
answer and one verdict per test in the suite. A batched call that returns a different
number of outputs (a lost batch, a truncated response) raises `BatchLengthError`
instead of scoring the remaining probes: an unchecked candidate is never accepted.
The same policy applies to proposals: a probe proposal the cumulative suite rejects,
like a memory proposal the memory rejects, costs that cycle its proposal and is
recorded in the trace, rather than aborting the cycle.

