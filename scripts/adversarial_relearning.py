"""
Adversarial Relearning Attack.

Tests whether unlearned models can re-acquire forgotten knowledge when
fine-tuned on a small portion of the forget set.

Strategies (from block diagram):
  - random_20: Random 20% of forget set
  - shortest_20: Shortest 20% of forget set (by token count)

Usage:
    python scripts/adversarial_relearning.py --model_path saves/unlearn/tofu_phi-1_5_forget05_SimNPO_SAM
    python scripts/adversarial_relearning.py --model_path saves/unlearn/tofu_phi-1_5_forget05_SimNPO_SAM --strategy shortest_20
    python scripts/adversarial_relearning.py --all --forget_split forget05
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

SAVES_DIR = ROOT / "saves"
UNLEARN_DIR = SAVES_DIR / "unlearn"

ALL_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "NPO_SAM", "SimNPO_SAM"]
MODEL = "phi-1_5"
STRATEGIES = ["random_20", "shortest_20"]

SPLIT_MAP = {
    "forget01": {"forget": "forget01", "retain": "retain99", "holdout": "holdout01"},
    "forget05": {"forget": "forget05", "retain": "retain95", "holdout": "holdout05"},
    "forget10": {"forget": "forget10", "retain": "retain90", "holdout": "holdout10"},
}


def run_relearning_attack(model_path, forget_split, strategy, epochs=3, lr=1e-5, batch_size=4):
    """Fine-tune the unlearned model on a subset of the forget set, then re-evaluate."""
    splits = SPLIT_MAP[forget_split]
    out_dir = Path(model_path) / f"relearn_{strategy}"

    if (out_dir / "RELEARN_DONE").exists():
        print(f"  [SKIP] {out_dir} already completed")
        return str(out_dir)

    print(f"  Relearning attack: {strategy} on {model_path}")

    # Use the framework's train.py with finetune trainer on the forget subset
    task_name = f"relearn_{Path(model_path).name}_{strategy}"
    cmd = [
        sys.executable, str(ROOT / "src" / "train.py"),
        "--config-name=train.yaml",
        f"model={MODEL}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        "trainer=finetune",
        f"task_name={task_name}",
        f"data=finetune",
        f"+data.train.args.hf_args.path=locuslab/tofu",
        f"+data.train.args.hf_args.name={splits['forget']}",
        f"trainer.args.num_train_epochs={epochs}",
        f"trainer.args.learning_rate={lr}",
        f"trainer.args.per_device_train_batch_size={batch_size}",
        f"trainer.args.output_dir={out_dir}",
        "trainer.args.save_strategy=no",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
    ]

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode == 0:
        (out_dir / "RELEARN_DONE").touch()
    return str(out_dir) if result.returncode == 0 else None


def run_eval_on_relearned(relearn_path, forget_split):
    """Evaluate the relearned model using TOFU metrics."""
    splits = SPLIT_MAP[forget_split]
    task_name = f"eval_{Path(relearn_path).parent.name}_{Path(relearn_path).name}"

    eval_output = Path(relearn_path) / "evals"
    if (eval_output / "TOFU_EVAL.json").exists():
        print(f"  [SKIP] Eval already exists for {relearn_path}")
        return

    cmd = [
        sys.executable, str(ROOT / "src" / "eval.py"),
        "experiment=eval/tofu/phi",
        f"forget_split={splits['forget']}",
        f"holdout_split={splits['holdout']}",
        f"model={MODEL}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={relearn_path}",
        f"paths.output_dir={eval_output}",
    ]

    retain_logs = SAVES_DIR / "eval" / f"tofu_{MODEL}_{splits['retain']}" / "TOFU_EVAL.json"
    if retain_logs.exists():
        cmd.append(f"retain_logs_path={retain_logs}")

    subprocess.run(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Adversarial Relearning Attack")
    parser.add_argument("--model_path", type=str, help="Path to unlearned model")
    parser.add_argument("--forget_split", default="forget05", choices=list(SPLIT_MAP.keys()))
    parser.add_argument("--strategy", default="random_20", choices=STRATEGIES)
    parser.add_argument("--epochs", type=int, default=3, help="Relearning epochs")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--all", action="store_true", help="Run on all unlearned models")
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    if args.all:
        for method in ALL_METHODS:
            model_path = UNLEARN_DIR / f"tofu_{MODEL}_{args.forget_split}_{method}"
            if not model_path.exists():
                print(f"[SKIP] {model_path} not found")
                continue
            for strategy in STRATEGIES:
                relearn_path = run_relearning_attack(
                    str(model_path), args.forget_split, strategy,
                    args.epochs, args.lr, args.batch_size
                )
                if relearn_path and not args.skip_eval:
                    run_eval_on_relearned(relearn_path, args.forget_split)
    else:
        if not args.model_path:
            parser.error("--model_path required when not using --all")
        relearn_path = run_relearning_attack(
            args.model_path, args.forget_split, args.strategy,
            args.epochs, args.lr, args.batch_size
        )
        if relearn_path and not args.skip_eval:
            run_eval_on_relearned(relearn_path, args.forget_split)


if __name__ == "__main__":
    main()
