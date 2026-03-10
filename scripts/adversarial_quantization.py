"""
Adversarial Quantization Attack.

Tests whether unlearning survives model quantization (FP16 -> 4-bit).
If forgotten knowledge reappears after quantization, the unlearning is fragile.

Usage:
    python scripts/adversarial_quantization.py --model_path saves/unlearn/tofu_phi-1_5_forget05_SimNPO_SAM
    python scripts/adversarial_quantization.py --all --forget_split forget05
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

SPLIT_MAP = {
    "forget01": {"forget": "forget01", "retain": "retain99", "holdout": "holdout01"},
    "forget05": {"forget": "forget05", "retain": "retain95", "holdout": "holdout05"},
    "forget10": {"forget": "forget10", "retain": "retain90", "holdout": "holdout10"},
}


def quantize_and_eval(model_path, forget_split):
    """Quantize the model to 4-bit and evaluate."""
    splits = SPLIT_MAP[forget_split]
    eval_output = Path(model_path) / "quant_4bit_evals"

    if (eval_output / "TOFU_EVAL.json").exists():
        print(f"  [SKIP] Quantized eval already exists for {model_path}")
        return

    print(f"  Quantization attack (4-bit): {model_path}")

    task_name = f"quant_{Path(model_path).name}"
    cmd = [
        sys.executable, str(ROOT / "src" / "eval.py"),
        "experiment=eval/tofu/phi",
        f"forget_split={splits['forget']}",
        f"holdout_split={splits['holdout']}",
        f"model={MODEL}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={model_path}",
        # Load in 4-bit quantization
        "+model.model_args.load_in_4bit=true",
        f"paths.output_dir={eval_output}",
    ]

    retain_logs = SAVES_DIR / "eval" / f"tofu_{MODEL}_{splits['retain']}" / "TOFU_EVAL.json"
    if retain_logs.exists():
        cmd.append(f"retain_logs_path={retain_logs}")

    subprocess.run(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Adversarial Quantization Attack")
    parser.add_argument("--model_path", type=str, help="Path to unlearned model")
    parser.add_argument("--forget_split", default="forget05", choices=list(SPLIT_MAP.keys()))
    parser.add_argument("--all", action="store_true", help="Run on all unlearned models")
    args = parser.parse_args()

    if args.all:
        for method in ALL_METHODS:
            model_path = UNLEARN_DIR / f"tofu_{MODEL}_{args.forget_split}_{method}"
            if not model_path.exists():
                print(f"[SKIP] {model_path} not found")
                continue
            quantize_and_eval(str(model_path), args.forget_split)
    else:
        if not args.model_path:
            parser.error("--model_path required when not using --all")
        quantize_and_eval(args.model_path, args.forget_split)


if __name__ == "__main__":
    main()
