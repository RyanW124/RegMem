"""Train one RegMem LoRA from a frozen base on prepared proposal or repair rows."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .proposer import apply_memory_operations, proposal_messages, repair_messages
from .typedefs import ConsolidationOutput, MemoryState, RegressionSuite
from .verifier import Verifier


@dataclass
class RewardCase:
    memory: MemoryState
    suite: RegressionSuite
    target_ids: set[str]
    prior_pass_ids: set[str]

    @classmethod
    def from_row(cls, row):
        from .grpo_data import memory_from_dict, suite_from_dict
        case = cls(memory_from_dict(row["memory"]), suite_from_dict(row["suite"]),
                   set(row["target_ids"]), set(row["prior_pass_ids"]))
        active = case.target_ids | case.prior_pass_ids
        ids = [test.id for test in case.suite.tests]
        if (not active or case.target_ids & case.prior_pass_ids
                or not active.issubset(ids) or len(ids) != len(set(ids))):
            raise ValueError("Invalid or empty probe categories in training row")
        if row["role"] == "repair" and (not case.target_ids or active != set(ids)):
            raise ValueError("A repair row must classify all probes and include a failure")
        return case


def probe_reward(role, target_ids, prior_pass_ids, passed):
    """Average group pass rates, renormalizing when a group has no members."""
    target_weight = {"proposal": 1, "repair": 2}[role]
    weighted_sum, total_weight = 0.0, 0
    for ids, weight in ((target_ids, target_weight), (prior_pass_ids, 1)):
        if ids:
            weighted_sum += weight * len(ids & passed) / len(ids)
            total_weight += weight
    if not total_weight:
        raise ValueError("Cannot score an empty probe suite")
    penalty = 1.25 if role == "repair" and prior_pass_ids - passed else 0.0
    return weighted_sum / total_weight - penalty


def score_completion(role, completion, case, verifier):
    """Score the sampled JSON. Only policy parse/apply errors get the invalid reward."""
    try:
        patch = ConsolidationOutput.model_validate_json(completion).operations
        candidate = apply_memory_operations(case.memory, patch)
    except ValueError:
        return -2.0
    results = verifier.verify(candidate, case.suite).results
    if [r.test_id for r in results] != [t.id for t in case.suite.tests]:
        raise RuntimeError("Verifier returned missing, repeated, or misordered probe results")
    passed = {r.test_id for r in results if r.passed}
    return probe_reward(role, case.target_ids, case.prior_pass_ids, passed)


def policy_prompt(row, case):
    """Refuse outdated prompt schemas, including repair rows containing a raw session."""
    saved = row["prompt"]
    payload = json.loads(saved[-1]["content"])
    if row["role"] == "repair":
        expected = repair_messages(case.memory, payload["failed_tests"], payload["previous_versions_of_changed_items"])
    else:
        expected = proposal_messages(case.memory, payload["new_evidence"])
    if saved != expected:
        raise ValueError("Saved policy prompt differs from runtime. Regenerate the training data")
    return expected


@contextmanager
def frozen_base(policy):
    """Scope reward inference so an exception also restores the training policy."""
    import torch
    was_training = policy.training
    try:
        policy.eval()
        with policy.disable_adapter(), torch.no_grad():
            yield
    finally:
        policy.train(was_training)


def load_split(path, role, tokenizer, row_limit, max_tokens, cases):
    from datasets import Dataset
    from .grpo_data import read_rows

    inputs, oversize, episodes = [], 0, set()
    for index, row in enumerate(read_rows(path)):
        if row["role"] != role:
            raise ValueError(f"{path}:{index + 1} belongs to {row['role']}")
        case = RewardCase.from_row(row)
        if role == "repair" and row.get("recoverable") is False:
            raise ValueError("A diagnostic unrecoverable failure entered the repair training split")
        # Render the same Qwen chat template as the runtime exactly once. A plain
        # prompt also makes TRL hand the original completion text to our parser.
        prompt = tokenizer.apply_chat_template(policy_prompt(row, case), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        if len(tokenizer(prompt)["input_ids"]) > max_tokens:
            oversize += 1
            continue
        case_id = f"{path.stem}:{index}"
        cases[case_id] = case
        inputs.append({"prompt": prompt, "case_id": case_id})
        episodes.add(str(row["episode_id"]))
        if len(inputs) == row_limit:
            break
    if not inputs:
        raise ValueError(f"No eligible rows in {path} ({oversize} exceeded {max_tokens} tokens)")
    print(f"{path.name}: {len(inputs)} selected, {oversize} over token budget", flush=True)
    return Dataset.from_list(inputs), episodes


def resumable_checkpoint(out):
    """Ignore a half-written latest save if the worker died during checkpointing."""
    checkpoints = [(int(p.name[11:]), p) for p in out.glob('checkpoint-*')
                   if p.is_dir() and p.name[11:].isdecimal()]
    for step, path in sorted(checkpoints, reverse=True):
        try:
            state = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
            files = ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors', 'adapter_config.json')
            if state.get('global_step') == step and all((path / f).stat().st_size > 0 for f in files):
                return str(path)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True, choices=["proposal", "repair"])
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path, help="Final PEFT adapter and numbered checkpoints")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision", help="Base model Hub revision, resolved to a commit when available")
    p.add_argument("--resume", action="store_true", help="Continue the latest checkpoint in --output-dir")
    p.add_argument("--resume-if-available", action="store_true",
                   help="Continue a complete checkpoint, or restart this run if it exited before saving one")
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--max-train-rows", type=int, default=128)
    p.add_argument("--max-dev-rows", type=int, default=32)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument("--scorer-max-tokens", type=int, default=256)
    p.add_argument("--scorer-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("max_steps", "max_dev_rows", "max_train_rows", "max_prompt_tokens",
                "max_completion_tokens", "scorer_batch_size", "scorer_max_tokens"):
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    if args.num_generations < 2:
        raise ValueError("GRPO needs at least two samples per prompt")

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.hub import cached_file, extract_commit_hash
    from trl import GRPOConfig, GRPOTrainer
    from .llm import LLM

    out = args.output_dir
    continuing = args.resume or args.resume_if_available
    manifest_path = out / "training_manifest.json"
    occupied = out.exists() and any(p.name != 'training_manifest.tmp' for p in out.iterdir())
    if occupied and (not continuing or not manifest_path.is_file()):
        raise FileExistsError(f"{out} is occupied. Resume a matching training run or choose a fresh directory")
    checkpoint = resumable_checkpoint(out) if continuing else None
    if args.resume and checkpoint is None:
        raise ValueError(f"No resumable checkpoint in {out}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GRPO training requires a CUDA GPU with BF16 support")

    config_file = cached_file(args.model, "config.json", revision=args.revision)
    revision = extract_commit_hash(config_file, None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cases, splits, episodes, hashes = {}, {}, {}, {}
    for split, limit in (("train", args.max_train_rows), ("dev", args.max_dev_rows)):
        path = args.data_dir / f"{args.role}_{split}.jsonl"
        splits[split], episodes[split] = load_split(path, args.role, tokenizer, limit,
                                                  args.max_prompt_tokens, cases)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    if episodes["train"] & episodes["dev"]:
        raise ValueError("Train and development selections share source episodes")
    split_manifest = args.data_dir / "episode_splits.json"
    if split_manifest.exists():
        hashes["episode_splits"] = hashlib.sha256(split_manifest.read_bytes()).hexdigest()

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model": args.model, "base_revision": revision, "role": args.role, "seed": args.seed,
                "rows": {k: len(v) for k, v in splits.items()}, "sha256": hashes,
                "num_generations": args.num_generations, "max_prompt_tokens": args.max_prompt_tokens,
                "max_completion_tokens": args.max_completion_tokens, "scorer_max_tokens": args.scorer_max_tokens,
                "scorer_batch_size": args.scorer_batch_size, "versions": {k: version(k) for k in
                    ("trl", "transformers", "torch", "peft", "accelerate", "datasets")}}
    if continuing and manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Checkpoint model, data, dependencies, or training configuration differs from this run")
    if not manifest_path.exists():
        tmp = manifest_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tmp.replace(manifest_path)
    if continuing:
        print(f"Training from {checkpoint or 'the base model (no complete checkpoint)'}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, dtype=torch.bfloat16)
    lora = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
                      revision=revision)
    interval = min(16, args.max_steps)
    train_args = GRPOConfig(
        output_dir=str(out), learning_rate=1e-5, max_steps=args.max_steps,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=1, gradient_accumulation_steps=args.num_generations,
        num_generations=args.num_generations, num_generations_eval=2, per_device_eval_batch_size=2,
        max_completion_length=args.max_completion_tokens, temperature=0.7, use_vllm=False,
        loss_type="grpo", beta=0.001, remove_unused_columns=False, seed=args.seed,
        save_strategy="steps", save_steps=interval, eval_strategy="steps", eval_steps=interval,
        load_best_model_at_end=True, metric_for_best_model="eval_reward", greater_is_better=True,
        report_to="none", logging_steps=1,
    )

    def regression_reward(completions, case_id, log_metric=None, **kwargs):
        with frozen_base(trainer.model):
            scores = [score_completion(args.role, text, cases[key], verifier)
                      for text, key in zip(completions, case_id, strict=True)]
        if log_metric:
            log_metric("valid_patch_fraction", sum(score != -2.0 for score in scores) / len(scores))
        return scores

    trainer = GRPOTrainer(model=base, args=train_args, reward_funcs=regression_reward, peft_config=lora,
                          train_dataset=splits["train"], eval_dataset=splits["dev"], processing_class=tokenizer)
    verifier = Verifier(LLM.from_model(trainer.model.get_base_model(), tokenizer, temperature=0.0,
                                       batch_size=args.scorer_batch_size, max_new_tokens=args.scorer_max_tokens))
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(out))
    trainer.save_state()
    trainer.save_metrics("train", {**result.metrics, "best_dev_reward": trainer.state.best_metric,
                                   "best_checkpoint": trainer.state.best_model_checkpoint})


if __name__ == "__main__":
    main()
