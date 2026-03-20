"""
Standalone visualization for Kaggle notebook cells.

Run this directly in a Kaggle notebook cell:
    %run scripts/visualize_notebook.py

Or copy the contents into a notebook cell. All data is loaded from the
saves/ directory automatically.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
ROOT = Path("/kaggle/working/MacUnlearn")
SAVES_DIR = ROOT / "saves"
UNLEARN_DIR = SAVES_DIR / "unlearn"
MODEL = "phi-1_5"
FORGET_SPLIT = "forget05"
ALL_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "NPO_SAM", "SimNPO_SAM"]

# ── Colors ─────────────────────────────────────────────────────────────────
COLORS = {
    "GradAscent":  "#e63946",
    "GradDiff":    "#457b9d",
    "NPO":         "#2a9d8f",
    "SimNPO":      "#e9c46a",
    "NPO_SAM":     "#264653",
    "SimNPO_SAM":  "#f4a261",
}
MARKERS = {"GradAscent": "o", "GradDiff": "s", "NPO": "D",
           "SimNPO": "^", "NPO_SAM": "v", "SimNPO_SAM": "P"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 13, "figure.dpi": 150, "savefig.dpi": 200,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

# ── Data loading ───────────────────────────────────────────────────────────
def load_eval(eval_dir):
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


def collect():
    results = {}
    for method in ALL_METHODS:
        task = f"tofu_{MODEL}_{FORGET_SPLIT}_{method}"
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


FIG_DIR = SAVES_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

results = collect()
methods = [m for m in ALL_METHODS if m in results and "standard" in results[m]]
print(f"Loaded results for {len(methods)} methods: {methods}")

# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Grouped Bar Chart: Metric Comparison
# ═══════════════════════════════════════════════════════════════════════════
metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE"]
metric_labels = ["Forget Quality", "Model Utility", "Forget QA Prob", "Forget QA ROUGE"]
n_methods = len(methods)
n_metrics = len(metrics)
bar_h = 0.14
y_pos = np.arange(n_methods)
palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261"]

fig1, ax = plt.subplots(figsize=(10, max(4, n_methods * 0.9 + 1)))
for i, (metric, lab) in enumerate(zip(metrics, metric_labels)):
    vals = [results[m]["standard"].get(metric, 0) for m in methods]
    offsets = y_pos + (i - n_metrics / 2 + 0.5) * bar_h
    bars = ax.barh(offsets, vals, height=bar_h, color=palette[i],
                   edgecolor="white", linewidth=0.5, label=lab)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=8.5, color="#333")

ax.set_yticks(y_pos)
ax.set_yticklabels(methods, fontsize=12, fontweight="medium")
ax.set_xlabel("Score")
ax.set_xlim(0, 1.05)
ax.set_title(f"Unlearning Method Comparison — TOFU {FORGET_SPLIT} (Phi‑1.5)",
             fontweight="bold", pad=12)
ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
ax.invert_yaxis()
fig1.tight_layout()
fig1.savefig(FIG_DIR / f"metric_comparison_{FORGET_SPLIT}.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Scatter: Forget Quality vs Model Utility Trade-off
# ═══════════════════════════════════════════════════════════════════════════
fig2, ax = plt.subplots(figsize=(8, 6))
for m in methods:
    fq = results[m]["standard"].get("forget_quality", 0)
    mu = results[m]["standard"].get("model_utility", 0)
    ax.scatter(mu, fq, s=180, c=COLORS[m], marker=MARKERS[m],
               edgecolors="white", linewidth=1.2, zorder=3)
    ax.annotate(m, (mu, fq), textcoords="offset points",
                xytext=(8, 6), fontsize=10, color="#333", fontweight="medium")

ax.axhspan(0.5, 1.05, color="#2a9d8f", alpha=0.06, zorder=0)
ax.axvspan(0.3, 0.55, color="#e9c46a", alpha=0.06, zorder=0)
ax.text(0.42, 0.96, "★ Ideal Region", fontsize=10, color="#2a9d8f",
        ha="center", fontstyle="italic", alpha=0.7)
ax.set_xlabel("Model Utility  →  higher is better")
ax.set_ylabel("Forget Quality  →  higher is better")
ax.set_xlim(-0.02, 0.55)
ax.set_ylim(-0.02, 1.05)
ax.set_title(f"Privacy–Utility Trade-off — TOFU {FORGET_SPLIT} (Phi‑1.5)",
             fontweight="bold", pad=12)
ax.grid(True, alpha=0.2, linestyle="--")
fig2.tight_layout()
fig2.savefig(FIG_DIR / f"tradeoff_scatter_{FORGET_SPLIT}.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Radar Chart: Method Profiles
# ═══════════════════════════════════════════════════════════════════════════
radar_metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob",
                 "forget_Q_A_ROUGE", "forget_Truth_Ratio"]
radar_labels  = ["Forget\nQuality", "Model\nUtility", "Forget\nQA Prob",
                 "Forget\nQA ROUGE", "Truth\nRatio"]

# Filter to available metrics
avail_m, avail_l = [], []
for met, lab in zip(radar_metrics, radar_labels):
    if any(results[m]["standard"].get(met) is not None for m in methods):
        avail_m.append(met)
        avail_l.append(lab)

if len(avail_m) >= 3:
    n = len(avail_m)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(avail_l, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#666")
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)

    for m in methods:
        vals = [results[m]["standard"].get(met, 0) for met in avail_m]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=COLORS[m], label=m)
        ax.fill(angles, vals, alpha=0.08, color=COLORS[m])

    ax.set_title(f"Method Profiles — TOFU {FORGET_SPLIT} (Phi‑1.5)",
                 fontweight="bold", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9, framealpha=0.9)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / f"radar_{FORGET_SPLIT}.png")
    plt.show()
else:
    print("⚠ Not enough metrics for radar chart (need ≥3)")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — SimNPO_SAM Relearning Attack Detail
# ═══════════════════════════════════════════════════════════════════════════
rl_metrics = ["forget_quality", "model_utility", "forget_Q_A_ROUGE", "extraction_strength"]
rl_labels  = ["Forget Quality", "Model Utility", "Forget QA\nROUGE", "Extraction\nStrength"]
before = [0.5453, 0.3898, 0.5639, 0.3627]
after  = [0.9647, 0.3849, 0.8391, 0.9803]

x = np.arange(len(rl_metrics))
bar_w = 0.32

fig4, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - bar_w / 2, before, bar_w, color="#264653",
            edgecolor="white", linewidth=0.8, label="Before Relearning")
b2 = ax.bar(x + bar_w / 2, after, bar_w, color="#e76f51",
            edgecolor="white", linewidth=0.8, label="After Relearning")

for bar_group in [b1, b2]:
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="medium")

for i, (b, a) in enumerate(zip(before, after)):
    delta = a - b
    color = "#2a9d8f" if delta > 0 else "#e63946"
    sign = "+" if delta > 0 else ""
    ax.annotate(f"{sign}{delta:.3f}", xy=(i, max(a, b) + 0.08),
                ha="center", fontsize=9, color=color, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(rl_labels, fontsize=11)
ax.set_ylim(0, 1.18)
ax.set_ylabel("Score")
ax.set_title("SimNPO + SAM — Relearning Attack Results\n(TOFU forget05, Phi‑1.5)",
             fontweight="bold", pad=12, fontsize=14)
ax.legend(fontsize=11, loc="upper left")
ax.grid(axis="y", alpha=0.15, linestyle="--")
fig4.tight_layout()
fig4.savefig(FIG_DIR / f"simnpo_sam_relearn_{FORGET_SPLIT}.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Heatmap: All Methods × All Metrics
# ═══════════════════════════════════════════════════════════════════════════
hm_metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE"]
hm_labels  = ["Forget Quality", "Model Utility", "Forget QA Prob", "Forget QA ROUGE"]

data_matrix = []
for m in methods:
    row = [results[m]["standard"].get(met, 0) for met in hm_metrics]
    data_matrix.append(row)
data_matrix = np.array(data_matrix)

fig5, ax = plt.subplots(figsize=(8, max(4, len(methods) * 0.7 + 1)))
im = ax.imshow(data_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(hm_labels)))
ax.set_xticklabels(hm_labels, fontsize=11, rotation=25, ha="right")
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=11, fontweight="medium")

# Annotate cells
for i in range(len(methods)):
    for j in range(len(hm_metrics)):
        val = data_matrix[i, j]
        text_color = "white" if val > 0.6 else "#333"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=10, fontweight="medium", color=text_color)

cbar = fig5.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Score", fontsize=11)
ax.set_title(f"Method × Metric Heatmap — TOFU {FORGET_SPLIT} (Phi‑1.5)",
             fontweight="bold", pad=12)
fig5.tight_layout()
fig5.savefig(FIG_DIR / f"heatmap_{FORGET_SPLIT}.png")
plt.show()

print(f"\n✓ All figures saved to {FIG_DIR}/")
