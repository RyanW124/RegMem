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
