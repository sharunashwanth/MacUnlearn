"""
Visualization for MacUnlearn experiment results (Kaggle notebook).

Usage:
    %run scripts/visualize_notebook.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
ROOT = Path("/kaggle/working/MacUnlearn")
SAVES_DIR = ROOT / "saves"
UNLEARN_DIR = SAVES_DIR / "unlearn"
MODEL = "phi-1_5"
FORGET_SPLIT = "forget05"
ALL_METHODS = ["GradAscent", "GradDiff", "NPO", "SimNPO", "NPO_SAM", "SimNPO_SAM"]

FIG_DIR = SAVES_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors & Style ─────────────────────────────────────────────────────────
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
    "font.size": 12, "axes.titlesize": 14,
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


results = collect()
methods = [m for m in ALL_METHODS if m in results and "standard" in results[m]]
print(f"Loaded {len(methods)} methods: {', '.join(methods)}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Privacy-Utility Trade-off Scatter
# ═══════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(9, 6.5))

# Shade ideal quadrant
ax.fill_between([0.3, 0.55], 0.5, 1.05, color="#2a9d8f", alpha=0.07, zorder=0)
ax.text(0.42, 0.97, "Ideal Region", fontsize=10, color="#2a9d8f",
        ha="center", fontstyle="italic", alpha=0.6)

for m in methods:
    fq = results[m]["standard"].get("forget_quality", 0)
    mu = results[m]["standard"].get("model_utility", 0)
    ax.scatter(mu, fq, s=220, c=COLORS[m], marker=MARKERS[m],
               edgecolors="white", linewidth=1.5, zorder=3, label=m)
    # Smart label offset to reduce overlap
    dx, dy = 10, 8
    if m == "NPO_SAM":
        dy = -14
    elif m == "NPO":
        dy = -14
    ax.annotate(m, (mu, fq), textcoords="offset points",
                xytext=(dx, dy), fontsize=10, color="#333", fontweight="medium")

ax.set_xlabel("Model Utility  (higher = better)")
ax.set_ylabel("Forget Quality  (higher = better)")
ax.set_xlim(-0.02, 0.55)
ax.set_ylim(-0.05, 1.08)
ax.set_title(f"Privacy-Utility Trade-off\nTOFU {FORGET_SPLIT} | Phi-1.5",
             fontweight="bold", pad=14)
ax.grid(True, alpha=0.15, linestyle="--")
fig1.tight_layout()
fig1.savefig(FIG_DIR / f"tradeoff_{FORGET_SPLIT}.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Heatmap: Methods x Metrics
# ═══════════════════════════════════════════════════════════════════════════
hm_metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE"]
hm_labels  = ["Forget Quality", "Model Utility", "Forget QA Prob", "Forget QA ROUGE"]

data_matrix = np.array([
    [results[m]["standard"].get(met, 0) for met in hm_metrics]
    for m in methods
])

fig2, ax = plt.subplots(figsize=(8, max(3.5, len(methods) * 0.65 + 1.2)))
im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(hm_labels)))
ax.set_xticklabels(hm_labels, fontsize=11, rotation=20, ha="right")
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods, fontsize=11, fontweight="medium")

for i in range(len(methods)):
    for j in range(len(hm_metrics)):
        val = data_matrix[i, j]
        text_color = "white" if val > 0.65 or val < 0.15 else "#222"
        ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=text_color)

cbar = fig2.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
cbar.set_label("Score", fontsize=11)
ax.set_title(f"Method Performance Overview\nTOFU {FORGET_SPLIT} | Phi-1.5",
             fontweight="bold", pad=14)
fig2.tight_layout()
fig2.savefig(FIG_DIR / f"heatmap_{FORGET_SPLIT}.png")
plt.show()


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Radar: Method Profiles
# ═══════════════════════════════════════════════════════════════════════════
radar_metrics = ["forget_quality", "model_utility", "forget_Q_A_Prob",
                 "forget_Q_A_ROUGE", "forget_Truth_Ratio"]
radar_labels  = ["Forget\nQuality", "Model\nUtility", "Forget\nQA Prob",
                 "Forget\nQA ROUGE", "Truth\nRatio"]

avail_m, avail_l = [], []
for met, lab in zip(radar_metrics, radar_labels):
    if any(results[m]["standard"].get(met) is not None for m in methods):
        avail_m.append(met)
        avail_l.append(lab)

if len(avail_m) >= 3:
    n = len(avail_m)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig3, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))
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
        ax.plot(angles, vals, linewidth=2.2, color=COLORS[m], label=m)
        ax.fill(angles, vals, alpha=0.06, color=COLORS[m])

    ax.set_title(f"Method Profiles\nTOFU {FORGET_SPLIT} | Phi-1.5",
                 fontweight="bold", pad=24, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=9, framealpha=0.9)
    fig3.tight_layout()
    fig3.savefig(FIG_DIR / f"radar_{FORGET_SPLIT}.png")
    plt.show()
else:
    print("Not enough metrics for radar chart (need >= 3)")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Relearning Attack: SimNPO vs SimNPO_SAM Side-by-Side
# ═══════════════════════════════════════════════════════════════════════════
rl_methods = ["SimNPO", "SimNPO_SAM"]
rl_metrics = ["forget_quality", "model_utility", "forget_Q_A_ROUGE", "extraction_strength"]
rl_labels  = ["Forget\nQuality", "Model\nUtility", "Forget QA\nROUGE", "Extraction\nStrength"]

# Try loading relearning data dynamically; fall back to hardcoded
def get_relearn_data(method):
    """Return (before_dict, after_dict) for a method, or None."""
    if method not in results:
        return None
    entry = results[method]
    if "standard" not in entry:
        return None
    relearn_key = next((k for k in entry if k.startswith("relearn_")), None)
    if relearn_key and entry.get(relearn_key):
        return entry["standard"], entry[relearn_key]
    return None

# Hardcoded fallback from user's output (in case eval JSONs don't have extraction_strength)
FALLBACK = {
    "SimNPO_SAM": {
        "before": {"forget_quality": 0.5453, "model_utility": 0.3898,
                   "forget_Q_A_ROUGE": 0.5639, "extraction_strength": 0.3627},
        "after":  {"forget_quality": 0.9647, "model_utility": 0.3849,
                   "forget_Q_A_ROUGE": 0.8391, "extraction_strength": 0.9803},
    },
}

relearn_data = {}
for m in rl_methods:
    loaded = get_relearn_data(m)
    if loaded:
        relearn_data[m] = {"before": loaded[0], "after": loaded[1]}
    elif m in FALLBACK:
        relearn_data[m] = FALLBACK[m]

available_rl_methods = [m for m in rl_methods if m in relearn_data]

if available_rl_methods:
    n_panels = len(available_rl_methods)
    fig4, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6),
                               sharey=True, squeeze=False)
    axes = axes[0]

    bar_w = 0.30
    x = np.arange(len(rl_metrics))
    c_before = "#264653"
    c_after  = "#e76f51"

    for idx, (m, ax) in enumerate(zip(available_rl_methods, axes)):
        bdata = relearn_data[m]["before"]
        adata = relearn_data[m]["after"]
        before_vals = [bdata.get(met, 0) for met in rl_metrics]
        after_vals  = [adata.get(met, 0) for met in rl_metrics]

        b1 = ax.bar(x - bar_w / 2, before_vals, bar_w, color=c_before,
                    edgecolor="white", linewidth=0.8, label="Before Relearning")
        b2 = ax.bar(x + bar_w / 2, after_vals, bar_w, color=c_after,
                    edgecolor="white", linewidth=0.8, label="After Relearning")

        # Value labels
        for bars in [b1, b2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="medium")

        # Delta annotations
        for i, (b, a) in enumerate(zip(before_vals, after_vals)):
            delta = a - b
            color = "#2a9d8f" if delta > 0 else "#e63946"
            sign = "+" if delta > 0 else ""
            y_pos = max(a, b) + 0.07
            ax.annotate(f"{sign}{delta:.3f}", xy=(i, y_pos),
                        ha="center", fontsize=9, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.8, lw=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(rl_labels, fontsize=10)
        ax.set_ylim(0, 1.22)
        ax.set_title(m, fontweight="bold", fontsize=14, pad=10,
                     color=COLORS.get(m, "#333"))
        ax.grid(axis="y", alpha=0.12, linestyle="--")
        if idx == 0:
            ax.set_ylabel("Score", fontsize=12)
            ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    fig4.suptitle(f"Relearning Attack Robustness\nTOFU {FORGET_SPLIT} | Phi-1.5",
                  fontweight="bold", fontsize=15, y=1.04)
    fig4.tight_layout()
    fig4.savefig(FIG_DIR / f"relearning_comparison_{FORGET_SPLIT}.png")
    plt.show()
else:
    print("No relearning data found for SimNPO / SimNPO_SAM")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Relearning: Metric-by-Metric Comparison (SimNPO vs SimNPO_SAM)
# ═══════════════════════════════════════════════════════════════════════════
if len(available_rl_methods) == 2:
    comp_metrics = ["forget_quality", "model_utility", "forget_Q_A_ROUGE", "extraction_strength"]
    comp_labels  = ["Forget Quality", "Model Utility", "Forget QA ROUGE", "Extraction Strength"]

    fig5, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    c_map = {"SimNPO": ("#e9c46a", "#c9a43a"), "SimNPO_SAM": ("#f4a261", "#d48241")}

    for ax, metric, label in zip(axes, comp_metrics, comp_labels):
        x_pos = np.arange(2)  # [Before, After]
        bar_w = 0.30

        for i, m in enumerate(available_rl_methods):
            bdata = relearn_data[m]["before"]
            adata = relearn_data[m]["after"]
            vals = [bdata.get(metric, 0), adata.get(metric, 0)]
            c1, c2 = c_map[m]
            offset = (i - 0.5) * bar_w
            bars = ax.bar(x_pos + offset, vals, bar_w, color=COLORS[m],
                          edgecolor="white", linewidth=0.8, label=m)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                        f"{h:.3f}", ha="center", fontsize=9, fontweight="medium")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["Before\nRelearning", "After\nRelearning"], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_title(label, fontweight="bold", fontsize=13)
        ax.grid(axis="y", alpha=0.12, linestyle="--")
        ax.legend(fontsize=9, loc="upper left")

    fig5.suptitle(f"SimNPO vs SimNPO+SAM: Relearning Robustness\nTOFU {FORGET_SPLIT} | Phi-1.5",
                  fontweight="bold", fontsize=15, y=1.02)
    fig5.tight_layout()
    fig5.savefig(FIG_DIR / f"relearning_detail_{FORGET_SPLIT}.png")
    plt.show()
elif available_rl_methods:
    print("Only 1 method has relearning data, skipping comparison grid")


print(f"\nAll figures saved to {FIG_DIR}/")
