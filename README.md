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

## Training the consolidation and repair adapters

The initial proposal uses prior memory and the current session. Repair receives
the candidate items, pre-update versions of changed or deleted items, and the
failed questions with the answers the candidate gave. Reference answers stay in
the verifier. The probe generator and verifier use the frozen base model.

From the RegMem checkout on the Slurm login node, activate the project venv and
stage the dependencies and data. Set `HF_HOME` to the same scratch cache for
setup and compute jobs.

```bash
export HF_HOME="$HOME/scratch/848n/hf_cache"
python -m pip install -r requirements-grpo.txt
hf download Qwen/Qwen3-8B
hf download YuWangX/Memalpha train.parquet --repo-type dataset --revision d59b425407f3f1b0e96797e0b817d45062d3f588 --local-dir data/memalpha
bash grpo_pilot.sh submit
```

The five Slurm jobs prepare proposal examples, train the proposal LoRA, mine its
failures, train an independent repair LoRA, and evaluate the adapters on held-out transitions. Each
uses one A100. `RUN_DIR` defaults to `runs/grpo`. Use a fresh directory for a
new run. Each adapter directory contains PEFT weights, training metrics, and
numbered checkpoints. After a timeout, failure, or interrupted submission, run
`bash grpo_pilot.sh submit` again with the same `RUN_DIR` and settings. The
launcher checks final outputs, reuses active jobs for this run, and submits the
unfinished stages. Slurm cancels jobs whose success dependency becomes impossible.
Training continues from the latest complete optimizer checkpoint (every 16 pilot
steps), or from step zero if none was saved. Interrupted data preparation, mining,
and evaluation restart at the beginning of their stage. Fix repeatable errors
using the failed job's `grpo-<stage>-<jobid>.out` before resubmitting.

Data preparation uses the released Mem-α training parquet. It splits source
episodes before deriving up to four consecutive transitions and writes the IDs
to `data/episode_splits.json` under the run directory. Natural repair failures
are supplemented with verified corruptions of previously retained items.
Each training run defaults to 128 training rows, 32 development rows, and 64
optimizer steps with four completions per prompt.

### Standalone MemConflict evaluation

Stage the benchmark data before running MemConflict separately.

```bash
mkdir -p data/memconflict
curl -fL https://raw.githubusercontent.com/TaoZhen1110/MemConflict/f3718daf08f090ce805edd09ea3984bf4a106935/Data/Step4_4.jsonl -o data/memconflict/Step4_4.jsonl
```

For a single MemConflict configuration, run `python run_memconflict.py --instance 0 --arm probes regmem`
and supply `--proposal-adapter` or `--repair-adapter` with the corresponding checkpoint paths.
Omitting a path selects the base model for that role. Add `--compare-adapters` to run the
base and each available adapter combination with shared probes. Use `--instances 0 1 2` to
run multiple histories.

The separate fixed-transition evaluation in `evaluate_grpo.py` tries both repair policies on the same failed
candidate for each initial policy. Each evaluation
writes paired outcomes to `comparisons.jsonl` and aggregate results to
`metrics.json`. Recoverability is a lower bound measured by trying executable
restorations and removals using repair-visible items and checking the full
frozen suite. A null rate denotes an empty denominator.
