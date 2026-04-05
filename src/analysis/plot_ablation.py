#!/usr/bin/env python3
"""
Generate paper-ready ablation study figures from logs/ablation_results.csv.

Figures produced (both .pdf and .png):
  1. accuracy_vs_params.{pdf,png} / fig4_ablation_scatter.{pdf,png}
     — scatter: QAT acc vs params, with DW1 as the only deployed-model callout
  2. width_ablation.{pdf,png}       — bar: W1/W2/W3/W5 QAT accuracy
  3. depth_ablation.{pdf,png}       — bar: W2/D1/D2/R1 QAT accuracy (no-residual depth)
  4. energy_tradeoff.{pdf,png}      — scatter: QAT acc vs cnn_time (only if column present)

Usage:
    python plot_ablation.py \\
        --csv ../../ai8x-training/logs/ablation_results.csv \\
        --out-dir plots/ablation
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 150,
})

PALETTE = {
    "W1": "#4c78a8",
    "W2": "#f58518",
    "W3": "#e45756",
    "D1": "#72b7b2",
    "D2": "#54a24b",
    "R1": "#b279a2",
    "W4": "#ff9da7",
    "W5": "#9d755d",
}


WIDTH_SWEEP = {"W1", "W2", "W3", "W5"}
DEPTH_SWEEP = {"D1", "D2", "DW2"}
RESIDUAL_VARIANTS = {"R1", "W4"}
DEPLOYED_MODEL = "DW1"


def _pct(val):
    """Convert 0-1 float to percentage string."""
    try:
        return f"{float(val)*100:.1f}%"
    except (ValueError, TypeError):
        return str(val)


def _parse_acc(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"[saved] {name}.pdf/.png → {out_dir}")


# ---------------------------------------------------------------------------
# Figure 1: Accuracy vs Parameters scatter
# ---------------------------------------------------------------------------

def fig_accuracy_vs_params(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    group_styles = [
        ("Width variants (W1–W5)", WIDTH_SWEEP, "o", "tab:blue", 90),
        ("Depth variants (D1, D2, DW2)", DEPTH_SWEEP, "s", "tab:green", 90),
        ("Residual variants (R1, W4)", RESIDUAL_VARIANTS, "^", "tab:orange", 100),
        ("Deployed model (DW1)", {DEPLOYED_MODEL}, "*", "tab:red", 220),
    ]

    for label, ids, marker, color, size in group_styles:
        subset = df[df["id"].isin(ids)].copy()
        if subset.empty:
            continue
        x = subset["params"].astype(float) / 1e3
        y = subset["qat_val_acc"].apply(_parse_acc) * 100
        ax.scatter(
            x,
            y,
            s=size,
            marker=marker,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.95,
            label=label,
            zorder=4 if DEPLOYED_MODEL in ids else 3,
        )
        for _, row in subset.iterrows():
            exp_id = row["id"]
            params = float(row["params"]) / 1e3
            qat_acc = _parse_acc(row["qat_val_acc"])
            if np.isnan(qat_acc):
                continue
            offset = (7, 3)
            weight = "bold" if exp_id == DEPLOYED_MODEL else "normal"
            ax.annotate(
                exp_id,
                (params, qat_acc * 100),
                textcoords="offset points",
                xytext=offset,
                fontsize=8,
                fontweight=weight,
                color="black" if exp_id == DEPLOYED_MODEL else color,
            )

    ax.set_xlabel("Parameters (thousands)", fontsize=11)
    ax.set_ylabel("QAT Validation Accuracy (%)", fontsize=11)
    ax.set_title("MicroBird Ablation Study: Accuracy vs. Model Size", fontsize=13, weight="bold")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.grid(True, linewidth=0.4, alpha=0.3)

    best_qat = df["qat_val_acc"].apply(_parse_acc).max() * 100
    if not np.isnan(best_qat):
        ax.axhline(best_qat, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.25)

    ax.legend(
        frameon=True,
        fontsize=8.5,
        loc="lower right",
        labelspacing=0.7,
        handletextpad=0.8,
        borderpad=0.4,
        scatterpoints=1,
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "accuracy_vs_params")
    save_fig(fig, out_dir, "fig4_ablation_scatter")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Width ablation bar chart
# ---------------------------------------------------------------------------

def fig_width_ablation(df: pd.DataFrame, out_dir: Path):
    ids = ["W1", "W2", "W3", "W5"]
    subset = df[df["id"].isin(ids)].set_index("id").reindex(ids)

    accs = [_parse_acc(subset.loc[i, "qat_val_acc"]) * 100
            if i in subset.index and not np.isnan(_parse_acc(subset.loc[i, "qat_val_acc"]))
            else 0.0
            for i in ids]
    labels = [f"{i}\n(ch={int(subset.loc[i, 'base_ch'])})" if i in subset.index else i
              for i in ids]
    colors = [PALETTE.get(i, "#888") for i in ids]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.6, width=0.55)

    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("QAT Val Accuracy (%)", fontsize=11)
    ax.set_title("Width Ablation (extra_blocks=0)", fontsize=12)
    ax.set_ylim(0, max(accs) * 1.15 + 5 if accs else 100)

    fig.tight_layout()
    save_fig(fig, out_dir, "width_ablation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Depth ablation bar chart
# ---------------------------------------------------------------------------

def fig_depth_ablation(df: pd.DataFrame, out_dir: Path):
    ids = ["W2", "D1", "D2", "R1"]
    subset = df[df["id"].isin(ids)].set_index("id").reindex(ids)

    accs = [_parse_acc(subset.loc[i, "qat_val_acc"]) * 100
            if i in subset.index and not np.isnan(_parse_acc(subset.loc[i, "qat_val_acc"]))
            else 0.0
            for i in ids]
    extra_blocks = [int(subset.loc[i, "extra_blocks"]) if i in subset.index else 0
                    for i in ids]
    labels = [f"{i}\n(+{eb} blk{'s' if eb!=1 else ''})" for i, eb in zip(ids, extra_blocks)]
    colors = [PALETTE.get(i, "#888") for i in ids]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.6, width=0.55)

    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("QAT Val Accuracy (%)", fontsize=11)
    ax.set_title("Depth Ablation (base_ch=16)", fontsize=12)
    ax.set_ylim(0, max(accs) * 1.15 + 5 if accs else 100)

    # Annotate R1 as having residual
    if "R1" in ids and subset.loc["R1", "residual"]:
        r1_pos = ids.index("R1")
        ax.text(r1_pos, accs[r1_pos] / 2, "residual", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

    fig.tight_layout()
    save_fig(fig, out_dir, "depth_ablation")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Energy / latency trade-off (optional)
# ---------------------------------------------------------------------------

def fig_energy_tradeoff(df: pd.DataFrame, out_dir: Path):
    if "cnn_time_us" not in df.columns:
        print("[info] cnn_time_us column not present — skipping energy trade-off plot")
        return

    valid = df[df["cnn_time_us"].notna() & df["qat_val_acc"].apply(
        lambda v: not np.isnan(_parse_acc(v)))]
    if len(valid) < 2:
        print("[info] Not enough cnn_time_us data points — skipping energy trade-off plot")
        return

    fig, ax = plt.subplots(figsize=(5.5, 4))

    for _, row in valid.iterrows():
        exp_id = row["id"]
        t_ms = float(row["cnn_time_us"]) / 1000.0
        acc = _parse_acc(row["qat_val_acc"]) * 100
        color = PALETTE.get(exp_id, "#888")
        ax.scatter(t_ms, acc, s=100, color=color,
                   edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(exp_id, (t_ms, acc),
                    textcoords="offset points", xytext=(6, 2), fontsize=8, color=color)

    ax.set_xlabel("CNN Inference Time (ms)", fontsize=11)
    ax.set_ylabel("QAT Val Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy–Energy Trade-off", fontsize=12)

    fig.tight_layout()
    save_fig(fig, out_dir, "energy_tradeoff")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="../../ai8x-training/logs/ablation_results.csv",
                   help="Path to ablation_results.csv")
    p.add_argument("--out-dir", default="plots/ablation",
                   help="Directory for output figures")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[info] Loaded {len(df)} rows from {csv_path}")
    print(df.to_string(index=False))

    out_dir = Path(args.out_dir)

    fig_accuracy_vs_params(df, out_dir)
    fig_width_ablation(df, out_dir)
    fig_depth_ablation(df, out_dir)
    fig_energy_tradeoff(df, out_dir)

    print(f"\nAll figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
