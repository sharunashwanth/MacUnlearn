"""
Visualization script for MacUnlearn experiment results.

Generates publication-quality charts from TOFU unlearning experiment data.

Usage:
    python scripts/visualize_results.py --forget_split forget05
    python scripts/visualize_results.py --forget_split forget05 --include_relearning
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
SAVES_DIR = ROOT / "saves"
UNLEARN_DIR = SAVES_DIR / "unlearn"
OUTPUT_DIR = SAVES_DIR / "figures"

ALL_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "NPO_SAM", "SimNPO_SAM"]
MODEL = "phi-1_5"

# ── Style ──────────────────────────────────────────────────────────────────
# Color palette — distinct, colorblind-friendly
METHOD_COLORS = {
    "GradAscent":  "#e63946",
    "GradDiff":    "#457b9d",
    "NPO":         "#2a9d8f",
    "SimNPO":      "#e9c46a",
    "NPO_SAM":     "#264653",
    "SimNPO_SAM":  "#f4a261",
}
METHOD_MARKERS = {
    "GradAscent":  "o",
    "GradDiff":    "s",
    "NPO":         "D",
    "SimNPO":      "^",
    "NPO_SAM":     "v",
    "SimNPO_SAM":  "P",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Data loading ───────────────────────────────────────────────────────────
def load_eval(eval_dir):
    """Load evaluation JSON from a directory."""
    for name in ("TOFU_SUMMARY.json", "TOFU_EVAL.json"):
        p = Path(eval_dir) / name
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if name == "TOFU_EVAL.json":
                return {k: v.get("agg_value") for k, v in data.items()
                        if isinstance(v, dict) and "agg_value" in v}
            return data
    return None


def collect(forget_split):
    """Collect standard + relearning results for every method."""
    results = {}
    for method in ALL_METHODS:
        task = f"tofu_{MODEL}_{forget_split}_{method}"
        base = UNLEARN_DIR / task
        if not base.exists():
            continue
        entry = {"method": method}
        std = load_eval(base / "evals")
        if std:
            entry["standard"] = std
        for strategy in ["random_20", "shortest_20"]:
            rl = load_eval(base / f"relearn_{strategy}" / "evals")
            if rl:
                entry[f"relearn_{strategy}"] = rl
        results[method] = entry
    return results


# ── Chart 1: Grouped bar chart — all methods, key metrics ─────────────────
def plot_metric_comparison(results, forget_split):
    """Grouped horizontal bar chart of key metrics across methods."""
    metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE"]
    metric_labels = {
        "forget_quality":    "Forget Quality",
        "model_utility":     "Model Utility",
        "forget_Q_A_Prob":   "Forget QA Prob",
        "forget_Q_A_ROUGE":  "Forget QA ROUGE",
    }
    methods = [m for m in ALL_METHODS if m in results and "standard" in results[m]]
    if not methods:
        return None

    n_methods = len(methods)
    n_metrics = len(metrics)
    bar_h = 0.14
    y_positions = np.arange(n_methods)

    fig, ax = plt.subplots(figsize=(10, max(4, n_methods * 0.9 + 1)))

    for i, metric in enumerate(metrics):
        vals = [results[m]["standard"].get(metric, 0) for m in methods]
        offsets = y_positions + (i - n_metrics / 2 + 0.5) * bar_h
        bars = ax.barh(offsets, vals, height=bar_h, color=_metric_color(i),
                       edgecolor="white", linewidth=0.5, label=metric_labels[metric])
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=8.5, color="#333")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(methods, fontsize=12, fontweight="medium")
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Unlearning Method Comparison — TOFU {forget_split} (Phi‑1.5)",
                 fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def _metric_color(idx):
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
    return palette[idx % len(palette)]


# ── Chart 2: Forget Quality vs Model Utility scatter ──────────────────────
def plot_tradeoff_scatter(results, forget_split):
    """Scatter plot showing the privacy–utility trade-off."""
    methods = [m for m in ALL_METHODS if m in results and "standard" in results[m]]
    if not methods:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    for m in methods:
        fq = results[m]["standard"].get("forget_quality", 0)
        mu = results[m]["standard"].get("model_utility", 0)
        ax.scatter(mu, fq, s=180, c=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                   edgecolors="white", linewidth=1.2, zorder=3)
        # Offset label to avoid overlap
        ax.annotate(m, (mu, fq), textcoords="offset points",
                    xytext=(8, 6), fontsize=10, color="#333",
                    fontweight="medium")

    # Ideal region
    ax.axhspan(0.5, 1.05, color="#2a9d8f", alpha=0.06, zorder=0)
    ax.axvspan(0.3, 0.55, color="#e9c46a", alpha=0.06, zorder=0)
    ax.text(0.42, 0.96, "* Ideal Region *", fontsize=10, color="#2a9d8f",
            ha="center", fontstyle="italic", alpha=0.7)

    ax.set_xlabel("Model Utility  →  higher is better", fontsize=12)
    ax.set_ylabel("Forget Quality  →  higher is better", fontsize=12)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Privacy–Utility Trade-off — TOFU {forget_split} (Phi‑1.5)",
                 fontweight="bold", pad=12)
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


# ── Chart 3: Radar/spider chart ───────────────────────────────────────────
def plot_radar(results, forget_split):
    """Radar chart comparing top methods across all available metrics."""
    metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob",
               "forget_Q_A_ROUGE", "forget_Truth_Ratio"]
    metric_labels = ["Forget\nQuality", "Model\nUtility", "Forget\nQA Prob",
                     "Forget\nQA ROUGE", "Truth\nRatio"]

    # Only include methods with standard results
    methods = [m for m in ALL_METHODS if m in results and "standard" in results[m]]
    if not methods:
        return None

    # Filter to metrics that exist in at least one method
    available = []
    available_labels = []
    for met, lab in zip(metrics, metric_labels):
        if any(results[m]["standard"].get(met) is not None for m in methods):
            available.append(met)
            available_labels.append(lab)
    if len(available) < 3:
        return None

    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666")
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

    for m in methods:
        vals = [results[m]["standard"].get(met, 0) for met in available]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=METHOD_COLORS[m], label=m)
        ax.fill(angles, vals, alpha=0.08, color=METHOD_COLORS[m])

    ax.set_title(f"Method Profiles — TOFU {forget_split} (Phi‑1.5)",
                 fontweight="bold", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12),
              fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Chart 4: Relearning robustness (before/after) ─────────────────────────
def plot_relearning(results, forget_split):
    """Grouped bar chart: before vs after relearning attack for each method."""
    metrics = ["forget_quality", "model_utility", "forget_Q_A_ROUGE"]
    metric_labels = ["Forget Quality", "Model Utility", "Forget QA ROUGE"]

    # Find methods that have relearning results
    methods_with_relearn = []
    for m in ALL_METHODS:
        if m in results and "standard" in results[m]:
            if any(k.startswith("relearn_") for k in results[m]):
                methods_with_relearn.append(m)

    if not methods_with_relearn:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    bar_w = 0.25
    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(methods_with_relearn))
        # Before (standard)
        before = [results[m]["standard"].get(metric, 0) for m in methods_with_relearn]
        # After (first available relearn strategy)
        after = []
        for m in methods_with_relearn:
            relearn_key = next((k for k in results[m] if k.startswith("relearn_")), None)
            after.append(results[m][relearn_key].get(metric, 0) if relearn_key else 0)

        ax.bar(x - bar_w / 2, before, bar_w, color="#264653", label="Before Relearn", edgecolor="white")
        ax.bar(x + bar_w / 2, after, bar_w, color="#e76f51", label="After Relearn", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_with_relearn, rotation=30, ha="right", fontsize=10)
        ax.set_title(label, fontweight="bold")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

        # Value labels
        for xi, (b, a) in enumerate(zip(before, after)):
            ax.text(xi - bar_w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
            ax.text(xi + bar_w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)

    axes[0].legend(fontsize=10, loc="upper left")
    fig.suptitle(f"Relearning Attack Robustness — TOFU {forget_split} (Phi‑1.5)",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ── Chart 5: SimNPO_SAM relearning detail (hardcoded data from user) ──────
def plot_simnpo_sam_relearning():
    """Detailed before/after chart for SimNPO_SAM relearning results."""
    metrics = ["forget_quality", "model_utility", "forget_Q_A_ROUGE", "extraction_strength"]
    labels  = ["Forget Quality", "Model Utility", "Forget QA\nROUGE", "Extraction\nStrength"]
    before  = [0.5453, 0.3898, 0.5639, 0.3627]
    after   = [0.9647, 0.3849, 0.8391, 0.9803]

    x = np.arange(len(metrics))
    bar_w = 0.32

    fig, ax = plt.subplots(figsize=(9, 5.5))

    b1 = ax.bar(x - bar_w / 2, before, bar_w, color="#264653",
                edgecolor="white", linewidth=0.8, label="Before Relearning")
    b2 = ax.bar(x + bar_w / 2, after, bar_w, color="#e76f51",
                edgecolor="white", linewidth=0.8, label="After Relearning")

    # Value labels
    for bar_group in [b1, b2]:
        for bar in bar_group:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="medium")

    # Delta arrows
    for i, (b, a) in enumerate(zip(before, after)):
        delta = a - b
        color = "#2a9d8f" if delta > 0 else "#e63946"
        sign = "+" if delta > 0 else ""
        ax.annotate(f"{sign}{delta:.3f}", xy=(i, max(a, b) + 0.08),
                    ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("SimNPO + SAM — Relearning Attack Results\n(TOFU forget05, Phi‑1.5)",
                 fontweight="bold", pad=12, fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.15, linestyle="--")
    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize unlearning results")
    parser.add_argument("--forget_split", default="forget05")
    parser.add_argument("--include_relearning", action="store_true",
                        help="Include relearning robustness chart")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    results = collect(args.forget_split)
    if not results:
        print(f"No results found for {args.forget_split}.")
        return

    charts = {
        "metric_comparison": plot_metric_comparison(results, args.forget_split),
        "tradeoff_scatter":  plot_tradeoff_scatter(results, args.forget_split),
        "radar":             plot_radar(results, args.forget_split),
        "simnpo_sam_relearn": plot_simnpo_sam_relearning(),
    }

    if args.include_relearning:
        charts["relearning_robustness"] = plot_relearning(results, args.forget_split)

    saved = []
    for name, fig in charts.items():
        if fig is None:
            continue
        path = out / f"{name}_{args.forget_split}.png"
        fig.savefig(path)
        plt.close(fig)
        saved.append(str(path))
        print(f"Saved: {path}")

    print(f"\n✓ {len(saved)} figures saved to {out}")


if __name__ == "__main__":
    main()
