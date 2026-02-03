#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Aggregate misalignment rates by domain/model from all_runs_summary.csv and 
produce a 2x2 grid plot with a shared legend."""

import argparse
import os
import string
import re
from math import ceil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

# ================= Style tuning (edit here) =================
BAR_WIDTH = 0.5         # Bar width (0.0 - 1.0), recommended 0.4 ~ 0.6.
FONT_SCALE = 35         # Base font size, recommended 12 ~ 18.
FIG_SIZE = (26, 18)     # Canvas size (width, height); increase with font size.
# ============================================================

# --- Global style settings ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # Prefer sans-serif fonts.
    'font.size': FONT_SCALE,
    'axes.titlesize': FONT_SCALE + 4,   # Title larger than body text.
    'axes.labelsize': FONT_SCALE + 2,   # Axis labels larger than body text.
    'xtick.labelsize': FONT_SCALE,
    'ytick.labelsize': FONT_SCALE,
    'legend.fontsize': FONT_SCALE,
    'figure.dpi': 200,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,              # Thicker axis lines.
})

# --- Color palette (Align on top, Strategic on bottom) ---
COLOR_MAP = {
    "behavior_align": "#BDE3C3",        # Blue (safest)
    "behavior_failattempt": "#A3CCDA",  # Yellow (attempt)
    "behavior_direct": "#F39EB6",       # Red (direct)
    "behavior_strategic": "#F5D2D2",    # Pink (strategic - most severe)
}
DEFAULT_COLORS = ["#CCB974", "#64B5CD", "#55A868"]

def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.96):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    p = np.divide(k, n, out=np.zeros_like(k), where=(n > 0))
    denom = 1.0 + (z**2) / np.where(n > 0, n, 1.0)
    center = (p + (z**2) / (2.0 * np.where(n > 0, n, 1.0))) / denom
    adj = (z / denom) * np.sqrt(
        np.divide(p * (1 - p), np.where(n > 0, n, 1.0), out=np.zeros_like(p), where=(n > 0))
        + (z**2) / (4.0 * (np.where(n > 0, n, 1.0) ** 2))
    )
    return center, (np.clip(center + adj, 0, 1) - center), np.clip(center - adj, 0, 1), np.clip(center + adj, 0, 1)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _short_model_label(name: str) -> str:
    lowered = (name or "").strip().lower()
    lowered = lowered.replace("-preview", "")
    lowered = re.sub(r"-0\\d{3,}$", "", lowered)
    lowered = re.sub(r"-\\d{8}$", "", lowered)
    replacements = {
        "gpt-5.1": "GPT-5.1",
        "gpt-5-mini": "GPT-5-mini",
        "gpt-4o": "GPT-4o",
        "llama-3.3": "Llama-3.3",
        "deepseek-r1": "Deepseek-R1",
        "deepseek-v3.2": "Deepseek-V3.2",
        "gemini-3-flash": "Gemini-3-Flash",
        "qwen3-235b-thinking": "Qwen-3-235B-Thinking",
        "qwen3-235b": "Qwen-3-235B",
        "qwen3-32b": "Qwen-3-32B",
    }
    for key, short in replacements.items():
        if lowered == key or lowered.startswith(key):
            return short
    return lowered


def plot_grid_distribution(df_agg: pd.DataFrame, outdir: Path, top_k: int | None = None):
    
    # Stack order: first entry is bottom (Direct bottom, Align top).
    stack_order = [
        "behavior_direct",
        "behavior_strategic",
        "behavior_failattempt",
        "behavior_align"       
    ]
    
    available_cols = [c for c in stack_order if c in df_agg.columns]
    extra_cols = [c for c in df_agg.columns if c.startswith("behavior_") and c not in available_cols]
    final_cols = available_cols[:2] + extra_cols + available_cols[2:] 

    domains = sorted(df_agg["domain"].dropna().unique().tolist())
    plot_domains = domains[:4] 

    # Create figure.
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharey=True)
    axes_flat = axes.flatten()

    legend_handles_map = {}

    def _index_label(idx: int) -> str:
        letters = string.ascii_lowercase
        if idx < len(letters):
            return letters[idx]
        idx -= len(letters)
        return letters[idx // len(letters)] + letters[idx % len(letters)]

    all_models = sorted(df_agg["model"].astype(str).unique().tolist())
    model_label_map = {name: _index_label(i) for i, name in enumerate(all_models)}

    for i, ax in enumerate(axes_flat):
        if i >= len(plot_domains):
            ax.axis('off')
            continue
            
        domain = plot_domains[i]
        
        sub = df_agg[df_agg["domain"] == domain].copy()
        sub = sub[sub["valid_runs"] > 0].copy()
        sub = sub.copy()
        sub["model_rank"] = sub["model"].map(
            {name: idx for idx, name in enumerate(all_models)}
        ).fillna(len(all_models))
        sub = sub.sort_values(["model_rank", "model"])

        if top_k and top_k > 0:
            sub = sub.head(top_k)

        denom = sub["valid_runs"].to_numpy().reshape(-1, 1)
        beh = sub[final_cols].to_numpy(dtype=float)
        beh_prop = np.divide(beh, denom, out=np.zeros_like(beh), where=(denom > 0))
        
        x = np.arange(len(sub))
        model_names = sub["model"].astype(str).tolist()
        labels = [model_label_map.get(name, "?") for name in model_names]

        # --- Draw stacked bars (thinner) ---
        bottom = np.zeros(len(sub))
        for j, col in enumerate(final_cols):
            color = COLOR_MAP.get(col, DEFAULT_COLORS[j % len(DEFAULT_COLORS)])
            label_name = col.replace("behavior_", "").capitalize()
            
            # Key tweak: width=BAR_WIDTH (0.5).
            bar = ax.bar(x, beh_prop[:, j], bottom=bottom, label=label_name,
                   color=color, edgecolor='white', linewidth=0.8, alpha=0.9, width=BAR_WIDTH)
            bottom += beh_prop[:, j]
            
            if label_name not in legend_handles_map:
                legend_handles_map[label_name] = bar

        # --- Error bars (more visible) ---
        _, _, lo, hi = wilson_ci(sub["misalign_num"].to_numpy(), sub["valid_runs"].to_numpy())
        y_rate = sub["misalign_rate"].to_numpy()
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
        yerr_low = np.maximum(y_rate - lo, 0.0)
        yerr_high = np.maximum(hi - y_rate, 0.0)
        yerr = np.vstack([yerr_low, yerr_high])
        
        # Thicker error bars.
        ax.errorbar(x, y_rate, yerr=yerr, fmt="none", 
                    ecolor="#222222", elinewidth=3.0, capsize=4, capthick=2.0, zorder=5)
        # Larger markers.
        ax.scatter(x, y_rate, s=80, facecolors='#222222', edgecolors='white', 
                   linewidth=3.0, zorder=6, label="Misalign Rate")
        
        if "Misalignment Rate" not in legend_handles_map:
             legend_handles_map["Misalignment Rate"] = Line2D([0], [0], color='#222222', marker='o', 
                                                              markerfacecolor='#222222', markeredgecolor='white',
                                                              linewidth=3.0, markersize=14)

        # --- Styling tweaks ---
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x)
        # X-axis labels: use letters, put model names in legend.
        ax.set_xticklabels(labels, rotation=0, ha="center", fontweight='medium')
        if i < 2:
            ax.tick_params(labelbottom=False)
        
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f"{domain}", fontweight='bold', pad=12)
        
        # Horizontal grid only, slightly darker.
        ax.grid(True, axis="y", linestyle='--', alpha=0.5, color='#aaaaaa')
        
        if i % 2 == 0:
            ax.set_ylabel("Proportion", fontweight='bold')

    # --- Global legend ---
    legend_keys = [c.replace("behavior_", "").capitalize() for c in reversed(final_cols)]
    legend_keys.append("Misalignment Rate")
    
    handles = [legend_handles_map[k] for k in legend_keys if k in legend_handles_map]
    labels = [k for k in legend_keys if k in legend_handles_map]

    # Larger legend font.
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.20),
               ncol=len(handles),
               frameon=False, prop={'weight': 'bold', 'size': FONT_SCALE + 2})

    model_handles = []
    for name, label in model_label_map.items():
        short_name = _short_model_label(name)
        model_handles.append(Line2D([], [], linestyle="none", marker="", label=f"{label}: {short_name}"))
    if model_handles:
        model_cols = max(1, ceil(len(model_handles) / 3))
        fig.legend(
            handles=model_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.06),
            frameon=False,
            fontsize=max(20, FONT_SCALE - 4),
            prop={'weight': 'bold'},
            ncol=model_cols,
            columnspacing=0.6,
            handletextpad=0.2,
            borderaxespad=0.2,
        )

    plt.tight_layout(rect=(0, 0.30, 1, 1))
    plt.subplots_adjust(bottom=0.30, hspace=0.25, wspace=0.15) # Increase spacing for large fonts.

    outpath = outdir / "combined_behavior_grid_v2.png"
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved plot: {outpath}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to all_runs_summary.csv")
    ap.add_argument("--outdir", type=str, default="outputs_misalign", help="Output directory")
    ap.add_argument("--topk", type=int, default=0, help="Keep top-K models per domain")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    df = pd.read_csv(csv_path)
    df = df[~df["model"].astype(str).str.lower().str.startswith("anthropic_")]

    if "avg_turn" in df.columns:
        df["avg_turn_sum"] = df["avg_turn"] * df["valid_runs"]
    else:
        df["avg_turn_sum"] = 0.0

    sum_cols = ["valid_runs", "misalign_num", "avg_turn_sum"] + [c for c in df.columns if c.startswith("behavior_")]
    df_agg = df.groupby(["domain", "model"], as_index=False)[sum_cols].sum()
    df_agg["avg_turn"] = np.divide(
        df_agg["avg_turn_sum"], df_agg["valid_runs"],
        out=np.zeros(len(df_agg), dtype=float),
        where=(df_agg["valid_runs"] > 0)
    )

    df_agg["misalign_rate"] = np.divide(
        df_agg["misalign_num"], df_agg["valid_runs"],
        out=np.zeros(len(df_agg), dtype=float),
        where=(df_agg["valid_runs"] > 0)
    )
    
    out_csv = outdir / "domain_model_misalign_summary.csv"
    df_agg.sort_values(["domain", "model"], ascending=[True, True]).to_csv(out_csv, index=False)

    top_k = None if args.topk <= 0 else args.topk
    plot_grid_distribution(df_agg, outdir, top_k=top_k)

if __name__ == "__main__":
    main()
