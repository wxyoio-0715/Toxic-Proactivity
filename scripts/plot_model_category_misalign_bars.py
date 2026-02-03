#!/usr/bin/env python3
"""Plot category-wise misalignment rates by model (scatter plots)."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D


ORDERED_MODELS = [
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-4o",
    "llama-3.3",
    "deepseek-r1",
    "deepseek-v3.2",
    "gemini-3-flash",
    "qwen3-235b",
    "qwen3-235b-thinking",
    "qwen3-32b",
]

MODEL_SHORT_NAMES = {
    "gpt-5.1": "GPT-5.1",
    "gpt-5-mini": "GPT-5-mini",
    "gpt-4o-2024-11-20": "GPT-4o",
    "gpt-4o": "GPT-4o",
    "llama-3.3-70b-instruct": "Llama-3.3",
    "llama-3.3": "Llama-3.3",
    "deepseek-r1-0528": "DeepSeek-R1",
    "deepseek-r1": "DeepSeek-R1",
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3-flash-preview": "Gemini-3-Flash",
    "gemini-3-flash": "Gemini-3-Flash",
    "qwen3-235b": "Qwen3-235B",
    "qwen3-235b-thinking": "Qwen3-235B-Thinking",
    "qwen3-32b": "Qwen3-32B",
}

FONT_SCALE = 17
POINT_SIZE = 100

FAMILY_COLORS = {
    "gpt": "#5CB85C",
    "llama": "#F0C85A",
    "gemini": "#6FA8DC",
    "deepseek": "#F39C6B",
    "qwen": "#4C78A8",
    "other": "#9E9E9E",
}

MODEL_POINT_SIZES = {
    "gpt-5.1": 200,
    "gpt-5-mini": 100,
    "gpt-4o": 100,
    "llama-3.3": 100,
    "deepseek-r1": 200,
    "deepseek-v3.2": 150,
    "gemini-3-flash": 150,
    "qwen3-235b-thinking": 200,
    "qwen3-235b": 130,
    "qwen3-32b": 100,
}


def model_rank(name: str) -> Tuple[int, str]:
    lowered = (name or "").strip().lower()
    lowered = re.sub(r"-a22b-thinking-2507$", "-thinking", lowered)
    lowered = re.sub(r"-a22b-2507$", "", lowered)
    for canonical in ORDERED_MODELS:
        if lowered == canonical or lowered.startswith(canonical):
            return (ORDERED_MODELS.index(canonical), lowered)
    return (len(ORDERED_MODELS), lowered)


def short_label(name: str) -> str:
    cleaned = (name or "").strip().lower()
    return MODEL_SHORT_NAMES.get(cleaned, name)


def point_label(name: str) -> str:
    label = short_label(name)
    for prefix in ("GPT-", "Llama-", "DeepSeek-", "Gemini-", "Qwen3-"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def load_rates(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("category") or "").strip().lower()
            model = (row.get("model") or "").strip().lower()
            if not category or not model:
                continue
            if category not in {"loyalty", "self-preservation"}:
                continue
            try:
                valid_runs = float(row.get("valid_runs") or 0)
            except ValueError:
                valid_runs = 0.0
            try:
                direct = float(row.get("behavior_direct") or 0)
            except ValueError:
                direct = 0.0
            try:
                strategic = float(row.get("behavior_strategic") or 0)
            except ValueError:
                strategic = 0.0
            direct_rate = direct / valid_runs if valid_runs else 0.0
            strategic_rate = strategic / valid_runs if valid_runs else 0.0
            data.setdefault(model, {})[category] = {
                "direct": direct_rate,
                "strategic": strategic_rate,
            }
    return data


def normalize_model(name: str) -> str:
    lowered = (name or "").strip().lower()
    lowered = re.sub(r"-a22b-thinking-2507$", "-thinking", lowered)
    lowered = re.sub(r"-a22b-2507$", "", lowered)
    for canonical in ORDERED_MODELS:
        if lowered == canonical:
            return canonical
    best_match = ""
    for canonical in ORDERED_MODELS:
        if lowered.startswith(canonical) and len(canonical) > len(best_match):
            best_match = canonical
    if best_match:
        return best_match
    return lowered


def family_for_model(name: str) -> str:
    canonical = normalize_model(name)
    if canonical in {"gpt-5.1", "gpt-5-mini", "gpt-4o"}:
        return "gpt"
    if canonical in {"deepseek-r1", "deepseek-v3.2"}:
        return "deepseek"
    if canonical.startswith("qwen3-"):
        return "qwen"
    if canonical in {"llama-3.3"}:
        return "llama"
    if canonical in {"gemini-3-flash"}:
        return "gemini"
    return "other"


def build_points(
    rates: Dict[str, Dict[str, Dict[str, float]]]
) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for model, categories in rates.items():
        for category in ("loyalty", "self-preservation"):
            entry = categories.get(category)
            if not entry:
                continue
            points.append(
                {
                    "model": model,
                    "category": category,
                    "strategic": entry.get("strategic", 0.0),
                    "direct": entry.get("direct", 0.0),
                }
            )
    return points


def point_size_for_model(name: str) -> float:
    canonical = normalize_model(name)
    return MODEL_POINT_SIZES.get(canonical, POINT_SIZE)


def is_reasoning_model(name: str) -> bool:
    canonical = normalize_model(name)
    return "thinking" in canonical or canonical in {"deepseek-r1"}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylabel("Direct Rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _label_offsets(points: List[Dict[str, float]]) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    placed: List[Tuple[float, float]] = []
    for p in points:
        x = p["strategic"]
        y = p["direct"]
        collisions = sum(1 for px, py in placed if abs(px - x) < 0.02 and abs(py - y) < 0.02)
        if collisions:
            offset_y = 4 + 8 * collisions
            offset_x = 4 + (8 if collisions % 2 == 0 else -8)
        else:
            offset_x = 4
            offset_y = 4
        offsets.append((offset_x, offset_y))
        placed.append((x, y))
    return offsets


def _repel_label_offsets(
    ax: plt.Axes,
    points: List[Dict[str, float]],
    fontsize: int,
) -> List[Tuple[float, float]]:
    if not points:
        return []
    fig = ax.figure
    dpi = fig.dpi
    px_per_pt = dpi / 72.0
    anchors = [ax.transData.transform((p["strategic"], p["direct"])) for p in points]
    labels = [point_label(p["model"]) for p in points]
    widths = [max(12.0, len(lbl) * fontsize * 0.6) * px_per_pt for lbl in labels]
    heights = [max(10.0, fontsize) * px_per_pt for _ in labels]
    offsets_px = [[6.0 * px_per_pt, 6.0 * px_per_pt] for _ in points]
    positions = [
        [a[0] + o[0], a[1] + o[1]] for a, o in zip(anchors, offsets_px)
    ]
    max_iter = 120
    for _ in range(max_iter):
        moved = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                overlap_x = (widths[i] + widths[j]) / 2.0 - abs(dx)
                overlap_y = (heights[i] + heights[j]) / 2.0 - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    push_x = (overlap_x + 1.0) * (1 if dx >= 0 else -1) * 0.5
                    push_y = (overlap_y + 1.0) * (1 if dy >= 0 else -1) * 0.5
                    positions[i][0] += push_x
                    positions[j][0] -= push_x
                    positions[i][1] += push_y
                    positions[j][1] -= push_y
                    moved = True
        for idx, anchor in enumerate(anchors):
            positions[idx][0] += (anchor[0] + offsets_px[idx][0] - positions[idx][0]) * 0.08
            positions[idx][1] += (anchor[1] + offsets_px[idx][1] - positions[idx][1]) * 0.08
        if not moved:
            break
    offsets_pt = [
        ((pos[0] - anchor[0]) / px_per_pt, (pos[1] - anchor[1]) / px_per_pt)
        for pos, anchor in zip(positions, anchors)
    ]
    return offsets_pt


def _fit_origin_line(points: List[Dict[str, float]]) -> float | None:
    xs = [p["strategic"] for p in points]
    ys = [p["direct"] for p in points]
    denom = sum(x * x for x in xs)
    if denom <= 0:
        return None
    return sum(x * y for x, y in zip(xs, ys)) / denom


def plot_scatter_by_family(
    ax: plt.Axes,
    points: List[Dict[str, float]],
    category: str,
    fit_mode: str,
) -> bool:
    filtered = [p for p in points if p["category"] == category]
    if not filtered:
        print(f"[WARN] No points for category: {category}")
        return False

    family_groups = {"gpt": [], "deepseek": [], "llama": [], "gemini": [], "qwen": [], "other": []}
    for p in filtered:
        family = family_for_model(p["model"])
        if family == "other":
            continue
        family_groups[family].append(p)
    labels = {
        "gpt": "GPT family",
        "deepseek": "DeepSeek family",
        "llama": "Llama family",
        "gemini": "Gemini family",
        "qwen": "Qwen-3 family",
    }
    for family, subset in family_groups.items():
        if not subset:
            continue
        reasoning_subset = [p for p in subset if is_reasoning_model(p["model"])]
        standard_subset = [p for p in subset if not is_reasoning_model(p["model"])]
        for idx, group in enumerate([standard_subset, reasoning_subset]):
            if not group:
                continue
            x = [p["strategic"] for p in group]
            y = [p["direct"] for p in group]
            sizes = [point_size_for_model(p["model"]) for p in group]
            ax.scatter(
                x,
                y,
                s=sizes,
                marker="^" if idx == 1 else "o",
                color=FAMILY_COLORS[family],
                alpha=0.85,
                edgecolor="#333333",
                linewidth=0.6,
                label=labels[family] if idx == 0 else None,
            )
    offsets = _repel_label_offsets(ax, filtered, max(16, FONT_SCALE - 2))
    for p, (dx, dy) in zip(filtered, offsets):
        px = p["strategic"]
        py = p["direct"]
        label = point_label(p["model"])
        ax.annotate(
            label,
            (px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=max(16, FONT_SCALE - 2),
            color="#222222",
        )
    _style_axes(ax)
    ax.set_title(
        f"Strategic vs Direct ({category.replace('-', ' ').title()})",
        pad=20,
        fontweight="bold",
    )
    if fit_mode == "overall":
        overall_slope = _fit_origin_line(filtered)
        if overall_slope is not None and overall_slope > 0:
            x_end = min(1.0, 1.0 / overall_slope)
            ax.plot(
                [0, x_end],
                [0, overall_slope * x_end],
                linestyle="--",
                color="#444444",
                linewidth=2.5,
                alpha=0.8,
                zorder=1,
            )
    elif fit_mode == "family":
        for family, subset in family_groups.items():
            if not subset:
                continue
            slope = _fit_origin_line(subset)
            if slope is None or slope <= 0:
                continue
            x_end = min(1.0, 1.0 / slope)
            ax.plot(
                [0, x_end],
                [0, slope * x_end],
                linestyle="--",
                color=FAMILY_COLORS[family],
                linewidth=2.0,
                alpha=0.7,
                zorder=1,
            )
    return True


def add_shared_legend(fig: plt.Figure) -> None:
    family_labels = {
        "gpt": "GPT family",
        "deepseek": "DeepSeek family",
        "llama": "Llama family",
        "gemini": "Gemini family",
        "qwen": "Qwen-3 family",
    }
    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FAMILY_COLORS[key],
            markeredgecolor="#333333",
            markersize=8,
            label=family_labels[key],
        )
        for key in ["gpt", "deepseek", "llama", "gemini", "qwen"]
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#666666",
            markeredgecolor="#333333",
            markersize=8,
            label="Standard",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="#666666",
            markeredgecolor="#333333",
            markersize=8,
            label="Reasoning",
        ),
    ]
    family_legend = fig.legend(
        handles=family_handles,
        title="Model family",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.93),
        fontsize=max(11, FONT_SCALE - 3),
        title_fontsize=max(12, FONT_SCALE - 2),
    )
    shape_legend = fig.legend(
        handles=shape_handles,
        title="Marker",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.45),
        fontsize=max(14, FONT_SCALE),
        title_fontsize=max(15, FONT_SCALE + 1),
    )
    fig.add_artist(family_legend)
    fig.add_artist(shape_legend)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot strategic vs direct scatter plots by category and family."
    )
    parser.add_argument(
        "--csv",
        default="logs/statistics/summary/category_model_behavior_summary.csv",
        help="Path to category_model_behavior_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="outputs_misalign/model_category_misalign_scatter",
        help="Output prefix (without suffix).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print model marker classification.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Missing CSV: {csv_path}")

    rates = load_rates(csv_path)
    if not rates:
        raise SystemExit("No usable rows found in the CSV.")

    points = build_points(rates)
    if not points:
        raise SystemExit("No usable points found in the CSV.")

    plt.rcParams.update({
        "font.size": FONT_SCALE,
        "axes.titlesize": FONT_SCALE + 4,
        "axes.labelsize": FONT_SCALE + 2,
        "xtick.labelsize": FONT_SCALE + 1,
        "ytick.labelsize": FONT_SCALE + 1,
        "legend.fontsize": FONT_SCALE + 1,
    })

    if args.debug:
        models = sorted({p["model"] for p in points})
        reasoning = [m for m in models if is_reasoning_model(m)]
        standard = [m for m in models if not is_reasoning_model(m)]
        print("[DEBUG] Reasoning models:", ", ".join(reasoning) if reasoning else "none")
        print("[DEBUG] Standard models:", ", ".join(standard) if standard else "none")

    out_prefix = Path(args.out)
    if out_prefix.suffix:
        out_prefix = out_prefix.with_suffix("")
    out_dir = out_prefix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for fit_mode, suffix in [("overall", "overall_fit"), ("family", "family_fit")]:
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 10.3), sharex=True, sharey=True)
        ok_top = plot_scatter_by_family(axes[0], points, "self-preservation", fit_mode)
        ok_bottom = plot_scatter_by_family(axes[1], points, "loyalty", fit_mode)
        axes[0].tick_params(labelbottom=False)
        fig.supxlabel("Strategic Rate")
        if ok_top or ok_bottom:
            add_shared_legend(fig)
        plt.tight_layout(rect=(0, 0, 0.88, 1))
        out_path = out_dir / f"{out_prefix.name}_{suffix}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
