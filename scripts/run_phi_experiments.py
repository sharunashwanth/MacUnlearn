"""
Main experiment runner for SimNPO+SAM research.
Runs all 6 unlearning methods on TOFU with Phi-1.5 with pause/resume support.

Usage:
    python scripts/run_phi_experiments.py                          # Run all methods
    python scripts/run_phi_experiments.py --methods NPO SimNPO     # Run specific methods
    python scripts/run_phi_experiments.py --forget_split forget01  # Change forget split
    python scripts/run_phi_experiments.py --skip_eval              # Skip evaluation
    python scripts/run_phi_experiments.py --resume                 # Resume from checkpoint

Each completed experiment is tracked in saves/unlearn/.completed_experiments.json
to enable pause/resume across runs.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.resolve()
SAVES_DIR = ROOT / "saves"
UNLEARN_DIR = SAVES_DIR / "unlearn"
EVAL_DIR = SAVES_DIR / "eval"
TRACKER_FILE = UNLEARN_DIR / ".completed_experiments.json"

# Split mapping
SPLIT_MAP = {
    "forget01": {"forget": "forget01", "retain": "retain99", "holdout": "holdout01"},
    "forget05": {"forget": "forget05", "retain": "retain95", "holdout": "holdout05"},
    "forget10": {"forget": "forget10", "retain": "retain90", "holdout": "holdout10"},
}

# All methods with their trainer names
ALL_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "NPO_SAM", "SimNPO_SAM"]

MODEL = "phi-1_5"
MODEL_PATH = "locuslab/tofu_ft_phi-1.5"


def load_tracker():
    if TRACKER_FILE.exists():
        with open(TRACKER_FILE, "r") as f:
            return json.load(f)
    return {"unlearned": [], "evaluated": []}


def save_tracker(tracker):
    TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKER_FILE, "w") as f:
        json.dump(tracker, f, indent=2)


def get_task_name(method, forget_split):
    return f"tofu_{MODEL}_{forget_split}_{method}"


def find_latest_checkpoint(task_name):
    """Find the latest checkpoint for resume."""
    task_dir = UNLEARN_DIR / task_name
    if not task_dir.exists():
        return None
    checkpoints = sorted(
        [d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def run_unlearn(method, forget_split, args, tracker, extra_args=[]):
    splits = SPLIT_MAP[forget_split]
    task_name = get_task_name(method, forget_split)

    if task_name in tracker["unlearned"] and not args.force:
        print(f"[SKIP] {task_name} already unlearned")
        return True

    print(f"\n{'='*60}")
    print(f"[UNLEARN] {task_name}: {method} on {forget_split}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/phi",
        f"trainer={method}",
        f"task_name={task_name}",
        f"model={MODEL}",
        f"model.model_args.pretrained_model_name_or_path={MODEL_PATH}",
        f"forget_split={splits['forget']}",
        f"retain_split={splits['retain']}",
        f"holdout_split={splits['holdout']}",
        f"trainer.args.per_device_train_batch_size={args.batch_size}",
        f"trainer.args.gradient_accumulation_steps={args.grad_accum}",
        "trainer.args.ddp_find_unused_parameters=true",
        "trainer.args.gradient_checkpointing=false",
        "trainer.args.eval_strategy=no",
        "trainer.args.eval_on_start=false",
        "+trainer.args.fp16=true",
        "trainer.args.bf16=false",
        "trainer.args.bf16_full_eval=false",
        "+trainer.args.fp16_full_eval=true",
        "trainer.args.optim=adamw_bnb_8bit",
        "+model.model_args.load_in_8bit=true",
        "+model.model_args.low_cpu_mem_usage=true",
    ] + extra_args

    # Resume from checkpoint if available
    checkpoint = find_latest_checkpoint(task_name) if args.resume else None
    if checkpoint:
        print(f"  Resuming from: {checkpoint}")
        cmd.append(f"+trainer.args.resume_from_checkpoint={checkpoint}")

    # Find pre-computed retain logs if we don't have our own yet
    retain_logs = None
    if EVAL_DIR.exists():
        # Look for any json in saves/eval that contains our split name AND model name
        for f in EVAL_DIR.rglob("*.json"):
            # Ensure it matches the split (e.g. retain95) AND the model (e.g. phi)
            if splits['retain'] in str(f) and "phi" in str(f).lower():
                retain_logs = f
                break
    
    if retain_logs:
        print(f"  Using pre-computed retain logs: {retain_logs.relative_to(ROOT) if retain_logs.is_relative_to(ROOT) else retain_logs}")
        cmd.append(f"eval.tofu.retain_logs_path={retain_logs}")
    else:
        print(f"  [WARN] No pre-computed retain logs found in {EVAL_DIR} for {splits['retain']} (phi)")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {task_name} failed with exit code {result.returncode}")
        return False

    tracker["unlearned"].append(task_name)
    save_tracker(tracker)
    print(f"[DONE] {task_name} unlearning complete")
    return True


def run_eval(method, forget_split, args, tracker, extra_args=[]):
    splits = SPLIT_MAP[forget_split]
    task_name = get_task_name(method, forget_split)

    if task_name in tracker["evaluated"] and not args.force:
        print(f"[SKIP] {task_name} already evaluated")
        return True

    model_path = UNLEARN_DIR / task_name
    if not model_path.exists():
        print(f"[SKIP] {task_name} model not found, skipping eval")
        return False

    print(f"\n{'='*60}")
    print(f"[EVAL] {task_name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(ROOT / "src" / "eval.py"),
        "experiment=eval/tofu/phi",
        f"forget_split={splits['forget']}",
        f"holdout_split={splits['holdout']}",
        f"model={MODEL}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        "trainer.args.bf16=false",
        "+model.model_args.low_cpu_mem_usage=true",
        "+model.model_args.load_in_4bit=true",
        f"paths.output_dir={model_path / 'evals'}",
    ]

    retain_logs = EVAL_DIR / f"tofu_{MODEL}_{splits['retain']}" / "TOFU_EVAL.json"
    if retain_logs.exists():
        cmd.append(f"retain_logs_path={retain_logs}")
    
    cmd += [arg for arg in extra_args if not arg.startswith('trainer.')]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"[FAIL] {task_name} eval failed with exit code {result.returncode}")
        return False

    tracker["evaluated"].append(task_name)
    save_tracker(tracker)
    print(f"[DONE] {task_name} evaluation complete")
    return True


def print_progress(tracker, methods, forget_split):
    """Print current progress inline (visible while cell is running)."""
    print(f"\n--- Progress ({len(tracker['unlearned'])}/{len(methods)} unlearned, "
          f"{len(tracker['evaluated'])}/{len(methods)} evaluated) ---")
    for m in methods:
        task = get_task_name(m, forget_split)
        u = "✅" if task in tracker["unlearned"] else "⬜"
        e = "✅" if task in tracker["evaluated"] else "⬜"
        print(f"  {u} unlearn | {e} eval  — {m}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run SimNPO+SAM experiments on TOFU with Phi-1.5")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS, help="Methods to run")
    parser.add_argument("--forget_split", default="forget05",
                        choices=list(SPLIT_MAP.keys()), help="Forget split to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after unlearning")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--force", action="store_true", help="Force re-run even if completed")
    parser.add_argument("--status", action="store_true", help="Show experiment status and exit")
    args, extra_args = parser.parse_known_args()

    # Initial tracker load
    tracker = load_tracker()

    if args.status:
        print_progress(tracker, ALL_METHODS, args.forget_split)
        return

    for method in args.methods:
        if not args.eval_only:
            success = run_unlearn(method, args.forget_split, args, tracker, extra_args)
            if not success:
                print(f"[WARN] {method} unlearning failed, skipping eval")
                print_progress(tracker, args.methods, args.forget_split)
                continue

        if not args.skip_eval:
            run_eval(method, args.forget_split, args, tracker, extra_args)

        print_progress(tracker, args.methods, args.forget_split)


    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print_progress(tracker, args.methods, args.forget_split)
    print(f"Results saved in: {UNLEARN_DIR}")


if __name__ == "__main__":
    main()
