"""
Comparative Analysis Script.

Aggregates evaluation results from all unlearning methods and produces
comparison tables and trade-off analysis.

Usage:
    python scripts/analysis.py --forget_split forget05
    python scripts/analysis.py --forget_split forget05 --include_adversarial
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.resolve()
SAVES_DIR = ROOT / "saves"
UNLEARN_DIR = SAVES_DIR / "unlearn"

ALL_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "NPO_SAM", "SimNPO_SAM"]
MODEL = "phi-1_5"

# Key metrics to compare
KEY_METRICS = [
    "forget_quality",
    "model_utility",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_Truth_Ratio",
]


def load_eval_results(eval_dir):
    """Load TOFU eval summary from a directory."""
    summary_file = Path(eval_dir) / "TOFU_SUMMARY.json"
    eval_file = Path(eval_dir) / "TOFU_EVAL.json"

    if summary_file.exists():
        with open(summary_file) as f:
            return json.load(f)
    elif eval_file.exists():
        with open(eval_file) as f:
            data = json.load(f)
        # Extract agg_values
        return {k: v.get("agg_value") for k, v in data.items() if isinstance(v, dict) and "agg_value" in v}
    return None


def collect_results(forget_split, include_adversarial=False):
    """Collect results from all methods."""
    results = {}

    for method in ALL_METHODS:
        task_name = f"tofu_{MODEL}_{forget_split}_{method}"
        base_dir = UNLEARN_DIR / task_name

        if not base_dir.exists():
            continue

        entry = {"method": method}

        # Standard eval
        eval_dir = base_dir / "evals"
        standard = load_eval_results(eval_dir)
        if standard:
            entry["standard"] = standard

        if include_adversarial:
            # Relearning attack results
            for strategy in ["random_20", "shortest_20"]:
                relearn_eval = base_dir / f"relearn_{strategy}" / "evals"
                relearn = load_eval_results(relearn_eval)
                if relearn:
                    entry[f"relearn_{strategy}"] = relearn

            # Quantization attack results
            quant_eval = base_dir / "quant_4bit_evals"
            quant = load_eval_results(quant_eval)
            if quant:
                entry["quant_4bit"] = quant

        results[method] = entry

    return results


def print_comparison_table(results, eval_type="standard"):
    """Print a formatted comparison table."""
    if not results:
        print("No results found.")
        return

    # Collect all available metrics
    available_metrics = set()
    for entry in results.values():
        if eval_type in entry:
            available_metrics.update(entry[eval_type].keys())

    metrics = [m for m in KEY_METRICS if m in available_metrics]
    if not metrics:
        metrics = sorted(available_metrics)

    # Header
    header = f"{'Method':<15}"
    for m in metrics:
        header += f"  {m:<20}"
    print(header)
    print("-" * len(header))

    # Rows
    for method in ALL_METHODS:
        if method not in results or eval_type not in results[method]:
            continue
        row = f"{method:<15}"
        for m in metrics:
            val = results[method][eval_type].get(m, "N/A")
            if isinstance(val, float):
                row += f"  {val:<20.4f}"
            else:
                row += f"  {str(val):<20}"
        print(row)


def save_results_json(results, output_path):
    """Save all results to a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comparative Analysis")
    parser.add_argument("--forget_split", default="forget05")
    parser.add_argument("--include_adversarial", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    results = collect_results(args.forget_split, args.include_adversarial)

    if not results:
        print(f"No results found for {args.forget_split}. Run experiments first:")
        print(f"  python scripts/run_phi_experiments.py --forget_split {args.forget_split}")
        return

    print(f"\n{'='*70}")
    print(f"  COMPARATIVE ANALYSIS — TOFU {args.forget_split} — Phi-1.5")
    print(f"{'='*70}")

    print(f"\n--- Standard Evaluation ---")
    print_comparison_table(results, "standard")

    if args.include_adversarial:
        print(f"\n--- Relearning Attack (Random 20%) ---")
        print_comparison_table(results, "relearn_random_20")

        print(f"\n--- Relearning Attack (Shortest 20%) ---")
        print_comparison_table(results, "relearn_shortest_20")

        print(f"\n--- Quantization Attack (4-bit) ---")
        print_comparison_table(results, "quant_4bit")

    output_path = args.output or str(SAVES_DIR / f"analysis_{MODEL}_{args.forget_split}.json")
    save_results_json(results, output_path)


if __name__ == "__main__":
    main()
