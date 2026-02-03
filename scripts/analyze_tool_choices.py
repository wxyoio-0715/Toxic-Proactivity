#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyze turn-level tool choices from turn-level CSV logs and
generate summary tables and plots."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


TOOL_COL_RE = re.compile(r"turn(\d+)_tool_name$")
FONT_SCALE = 22
LINE_PALETTE = [
    "#C5E1A5",
    "#A5D6A7",
    "#80CBC4",
    "#F8BBD0",
    "#E1BEE7",
    "#BA68C8",
]
LINE_WIDTH = 3.0
MARKER_SIZE = 8
TRIANGLE_ACTIONS = {"4", "5", "6"}
SCENARIO_SUFFIX_RE = re.compile(r"_\d+$")


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    n_safe = np.where(n > 0, n, 1.0)
    p = np.divide(k, n_safe, out=np.zeros_like(k), where=(n > 0))
    denom = 1.0 + (z ** 2) / n_safe
    center = (p + (z ** 2) / (2.0 * n_safe)) / denom
    margin = (z / denom) * np.sqrt(
        np.divide(p * (1 - p), n_safe, out=np.zeros_like(p), where=(n > 0))
        + (z ** 2) / (4.0 * (n_safe ** 2))
    )
    low = np.clip(center - margin, 0.0, 1.0)
    high = np.clip(center + margin, 0.0, 1.0)
    low = np.where(n > 0, low, 0.0)
    high = np.where(n > 0, high, 0.0)
    return low, high


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalize_tool_id(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return ""
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return text
    except ValueError:
        return text


def _normalize_scenario_name(name: object) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    text = str(name).strip()
    if not text:
        return ""
    return SCENARIO_SUFFIX_RE.sub("", text)


def _extract_turn_columns(columns: Iterable[str]) -> List[Tuple[int, str, str | None]]:
    turn_cols = []
    for col in columns:
        match = TOOL_COL_RE.match(col)
        if not match:
            continue
        turn_id = int(match.group(1))
        id_col = f"turn{turn_id}_tool_id"
        turn_cols.append((turn_id, col, id_col if id_col in columns else None))
    return sorted(turn_cols, key=lambda x: x[0])


def _tool_name_from_row(
    row: pd.Series,
    tool_col: str,
    id_col: str | None,
    label_mode: str,
    allowed_ids: set[str],
) -> str:
    tool = ""
    if label_mode == "id":
        if id_col:
            tool_id = _normalize_tool_id(row.get(id_col))
            tool = tool_id
    else:
        tool = _normalize_text(row.get(tool_col))
    if not tool and id_col:
        tool_id = _normalize_tool_id(row.get(id_col))
        if tool_id:
            tool = tool_id if label_mode == "id" else f"id_{tool_id}"
    if label_mode == "id" and tool and allowed_ids and tool not in allowed_ids:
        return ""
    return tool


def _build_long_df(
    df: pd.DataFrame,
    turn_cols: List[Tuple[int, str, str | None]],
    include_none: bool,
    label_mode: str,
    allowed_ids: set[str],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        model = row.get("model", "")
        scenario = row.get("scenario", "")
        run_file = row.get("run_file", "")
        for turn_id, tool_col, id_col in turn_cols:
            tool = _tool_name_from_row(
                row, tool_col, id_col, label_mode, allowed_ids
            )
            if not tool:
                if include_none:
                    tool = "none"
                else:
                    continue
            records.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "run_file": run_file,
                    "turn": turn_id,
                    "tool_name": tool,
                }
            )
    return pd.DataFrame(
        records,
        columns=["model", "scenario", "run_file", "turn", "tool_name"],
    )


def _add_all_models(df: pd.DataFrame) -> pd.DataFrame:
    df_all = df.copy()
    df_all["model"] = "ALL"
    return pd.concat([df, df_all], ignore_index=True)


def compute_scenario_tool_share(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        long_df.groupby(["model", "scenario", "tool_name"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        grouped.groupby(["model", "scenario"])["count"]
        .sum()
        .reset_index(name="total")
    )
    return grouped.merge(totals, on=["model", "scenario"], how="left")


def compute_turn_tool_share(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        long_df.groupby(["model", "turn", "tool_name"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        grouped.groupby(["model", "turn"])["count"]
        .sum()
        .reset_index(name="total")
    )
    return grouped.merge(totals, on=["model", "turn"], how="left")


def compute_trajectory_summary(
    df: pd.DataFrame,
    turn_cols: List[Tuple[int, str, str | None]],
    label_mode: str,
    allowed_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        tools: List[str] = []
        for _, tool_col, id_col in turn_cols:
            tool = _tool_name_from_row(
                row, tool_col, id_col, label_mode, allowed_ids
            )
            if not tool:
                break
            tools.append(tool)
        trajectory = " -> ".join(tools) if tools else "none"
        rows.append(
            {
                "model": row.get("model", ""),
                "scenario": row.get("scenario", ""),
                "trajectory": trajectory,
                "trajectory_len": len(tools),
            }
        )
    traj_df = pd.DataFrame(rows)
    counts = (
        traj_df.groupby(["model", "scenario", "trajectory", "trajectory_len"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        counts.groupby(["model", "scenario"])["count"]
        .sum()
        .reset_index(name="total")
    )
    out = counts.merge(totals, on=["model", "scenario"], how="left")
    return out


def plot_turn_share(
    df_turn: pd.DataFrame,
    outdir: Path,
    model: str,
    top_tools: int,
) -> None:
    sub = df_turn[df_turn["model"] == model].copy()
    if sub.empty:
        print(f"[WARN] No turn-level data for model '{model}'")
        return

    totals = sub.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = totals.head(top_tools).index.tolist()
    sub["tool_group"] = sub["tool_name"].where(
        sub["tool_name"].isin(keep_tools), other="other"
    )
    grouped = (
        sub.groupby(["turn", "tool_group"])["count"]
        .sum()
        .reset_index()
    )
    totals_turn = grouped.groupby("turn")["count"].sum().reset_index(name="total")
    grouped = grouped.merge(totals_turn, on="turn", how="left")
    grouped["pct"] = grouped["count"] / grouped["total"]
    pivot = grouped.pivot(index="turn", columns="tool_group", values="pct").fillna(0.0)
    pivot_count = grouped.pivot(index="turn", columns="tool_group", values="count").fillna(0.0)
    totals_by_turn = totals_turn.set_index("turn")["total"]

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    x_vals = pivot.index.to_numpy()
    for idx, tool in enumerate(pivot.columns):
        color = LINE_PALETTE[idx % len(LINE_PALETTE)]
        y_vals = pivot[tool].to_numpy()
        k_vals = pivot_count[tool].to_numpy()
        n_vals = totals_by_turn.reindex(pivot.index).to_numpy()
        low, high = wilson_interval(k_vals, n_vals)
        ax.fill_between(x_vals, low, high, color=color, alpha=0.2, linewidth=0)
        ax.plot(
            x_vals,
            y_vals,
            marker="^" if str(tool) in TRIANGLE_ACTIONS else "o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=color,
            label=tool,
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Turn")
    plt.ylabel("Tool Share")
    plt.title(f"Tool Share by Turn ({model})", fontsize=FONT_SCALE + 2, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=FONT_SCALE - 2)
    outpath = outdir / f"tool_share_by_turn_{model}.png"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {outpath}")


def _sort_tool_labels(labels: List[str]) -> List[str]:
    def _key(label: str) -> Tuple[int, str]:
        text = str(label)
        if text.isdigit():
            return (0, f"{int(text):03d}")
        return (1, text)

    return sorted(labels, key=_key)


def plot_turn_heatmap(
    df_turn: pd.DataFrame,
    outdir: Path,
    model: str,
    top_tools: int,
) -> None:
    sub = df_turn[df_turn["model"] == model].copy()
    if sub.empty:
        print(f"[WARN] No turn-level data for model '{model}'")
        return

    tool_totals = sub.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = tool_totals.head(top_tools).index.tolist()
    sub = sub[sub["tool_name"].isin(keep_tools)].copy()

    pivot = sub.pivot(index="turn", columns="tool_name", values="count").fillna(0.0)
    tool_order = _sort_tool_labels(pivot.columns.tolist())
    pivot = pivot[tool_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xlabel("Tool ID")
    ax.set_ylabel("Turn")
    ax.set_title(f"Tool Usage Heatmap ({model})", fontsize=FONT_SCALE + 2, fontweight="bold")
    ax.set_xticks(range(len(tool_order)))
    ax.set_xticklabels(tool_order, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    outpath = outdir / f"heatmap_tool_by_turn_{model}.png"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {outpath}")


def _plot_sankey(
    nodes: List[str],
    sources: List[int],
    targets: List[int],
    values: List[int],
    outpath: Path,
    title: str,
) -> None:
    print(f"[INFO] Sankey disabled. Skipping {outpath.name}.")


def plot_scenario_tool_sankey(
    df_scenario: pd.DataFrame,
    outdir: Path,
    model: str,
    top_scenarios: int,
    top_tools: int,
) -> None:
    sub = df_scenario[df_scenario["model"] == model].copy()
    if sub.empty:
        print(f"[WARN] No scenario-level data for model '{model}'")
        return

    scenario_names = sorted(
        sub["scenario"].dropna().astype(str).unique().tolist(),
        key=lambda s: s.lower(),
    )
    keep_scenarios = scenario_names[:top_scenarios]
    sub = sub[sub["scenario"].isin(keep_scenarios)]

    tool_totals = sub.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = tool_totals.head(top_tools).index.tolist()
    sub["tool_group"] = sub["tool_name"].where(
        sub["tool_name"].isin(keep_tools), other="other"
    )
    sub = (
        sub.groupby(["scenario", "tool_group"])["count"]
        .sum()
        .reset_index()
    )

    tool_order = (
        sub.groupby("tool_group")["count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    nodes = keep_scenarios + tool_order
    node_index = {name: idx for idx, name in enumerate(nodes)}
    sources = [node_index[row.scenario] for row in sub.itertuples()]
    targets = [node_index[row.tool_group] for row in sub.itertuples()]
    values = [int(row.count) for row in sub.itertuples()]

    outpath = outdir / f"sankey_scenario_tool_{model}.html"
    _plot_sankey(nodes, sources, targets, values, outpath, f"Scenario → Tool ({model})")


def plot_scenario_tool_lines(
    df_scenario: pd.DataFrame,
    outdir: Path,
    model: str,
    top_scenarios: int,
    top_tools: int,
) -> None:
    sub = df_scenario[df_scenario["model"] == model].copy()
    if sub.empty:
        print(f"[WARN] No scenario-level data for model '{model}'")
        return

    scenario_names = sorted(
        sub["scenario"].dropna().astype(str).unique().tolist(),
        key=lambda s: s.lower(),
    )
    keep_scenarios = scenario_names[:top_scenarios]
    sub = sub[sub["scenario"].isin(keep_scenarios)]

    tool_totals = sub.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = tool_totals.head(top_tools).index.tolist()
    sub["tool_group"] = sub["tool_name"].where(
        sub["tool_name"].isin(keep_tools), other="other"
    )
    grouped = (
        sub.groupby(["scenario", "tool_group"])["count"]
        .sum()
        .reset_index()
    )
    totals = grouped.groupby("scenario")["count"].sum().reset_index(name="total")
    grouped = grouped.merge(totals, on="scenario", how="left")
    grouped["pct"] = grouped["count"] / grouped["total"]
    pivot = grouped.pivot(index="scenario", columns="tool_group", values="pct").fillna(0.0)
    pivot_count = grouped.pivot(index="scenario", columns="tool_group", values="count").fillna(0.0)
    totals_by_scenario = totals.set_index("scenario")["total"]
    pivot = pivot.reindex(keep_scenarios)
    pivot_count = pivot_count.reindex(keep_scenarios)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    x = np.arange(len(pivot.index))
    for idx, tool in enumerate(pivot.columns):
        color = LINE_PALETTE[idx % len(LINE_PALETTE)]
        y_vals = pivot[tool].to_numpy()
        k_vals = pivot_count[tool].to_numpy()
        n_vals = totals_by_scenario.reindex(pivot.index).to_numpy()
        low, high = wilson_interval(k_vals, n_vals)
        ax.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
        ax.plot(
            x,
            y_vals,
            marker="^" if str(tool) in TRIANGLE_ACTIONS else "o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=color,
            label=tool,
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(x, pivot.index, rotation=30, ha="right")
    plt.xlabel("Scenario")
    plt.ylabel("Tool Share")
    plt.title(f"Scenario → Tool Share ({model})", fontsize=FONT_SCALE + 2, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=FONT_SCALE - 2)
    outpath = outdir / f"line_scenario_tool_{model}.png"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {outpath}")


def plot_scenario_tool_heatmap(
    df_scenario: pd.DataFrame,
    outdir: Path,
    model: str,
    top_scenarios: int,
    top_tools: int,
) -> None:
    sub = df_scenario[df_scenario["model"] == model].copy()
    if sub.empty:
        print(f"[WARN] No scenario-level data for model '{model}'")
        return

    scenario_names = sorted(
        sub["scenario"].dropna().astype(str).unique().tolist(),
        key=lambda s: s.lower(),
    )
    keep_scenarios = scenario_names[:top_scenarios]
    sub = sub[sub["scenario"].isin(keep_scenarios)]

    tool_totals = sub.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = tool_totals.head(top_tools).index.tolist()
    sub["tool_group"] = sub["tool_name"].where(
        sub["tool_name"].isin(keep_tools), other="other"
    )
    grouped = (
        sub.groupby(["scenario", "tool_group"])["count"]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot(index="scenario", columns="tool_group", values="count").fillna(0.0)
    pivot = pivot.reindex(keep_scenarios)
    tool_order = _sort_tool_labels(pivot.columns.tolist())
    pivot = pivot[tool_order]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xlabel("Tool ID")
    ax.set_ylabel("Scenario")
    ax.set_title(f"Scenario → Tool Count Heatmap ({model})", fontsize=FONT_SCALE + 2, fontweight="bold")
    ax.set_xticks(range(len(tool_order)))
    ax.set_xticklabels(tool_order, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    outpath = outdir / f"heatmap_scenario_tool_{model}.png"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {outpath}")


def plot_trajectory_sankey(
    df: pd.DataFrame,
    turn_cols: List[Tuple[int, str, str | None]],
    outdir: Path,
    scenario: str,
    model: str,
    top_tools: int,
    label_mode: str,
    all_models: List[str],
    allowed_ids: set[str],
) -> None:
    if model == "ALL":
        sub = df[df["scenario"] == scenario]
        if sub.empty:
            print(f"[WARN] No trajectory data for {scenario} (ALL)")
            return
        available = sorted(sub["model"].dropna().astype(str).unique().tolist())
        missing = sorted(set(all_models) - set(available))
        if missing:
            print(
                f"[WARN] {scenario} missing models ({len(missing)}): "
                + ", ".join(missing)
            )
    else:
        sub = df[(df["scenario"] == scenario) & (df["model"] == model)]
        if sub.empty:
            available = (
                df[df["scenario"] == scenario]["model"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            available_msg = ", ".join(sorted(available)) if available else "none"
            print(
                f"[WARN] No trajectory data for {scenario} ({model}). "
                f"Available models: {available_msg}"
            )
            return

    tool_counter: Counter[str] = Counter()
    transitions: Dict[Tuple[str, str], int] = defaultdict(int)

    for _, row in sub.iterrows():
        prev_label = None
        prev_turn = None
        for turn_id, tool_col, id_col in turn_cols:
            tool = _tool_name_from_row(
                row, tool_col, id_col, label_mode, allowed_ids
            )
            if not tool:
                break
            tool_counter[tool] += 1
            label = f"T{turn_id}:{tool}"
            if prev_label is not None and prev_turn is not None:
                transitions[(prev_label, label)] += 1
            prev_label = label
            prev_turn = turn_id

    keep_tools = {name for name, _ in tool_counter.most_common(top_tools)}
    filtered_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
    for (src, tgt), count in transitions.items():
        src_tool = src.split(":", 1)[1]
        tgt_tool = tgt.split(":", 1)[1]
        if src_tool not in keep_tools:
            src = f"{src.split(':', 1)[0]}:other"
        if tgt_tool not in keep_tools:
            tgt = f"{tgt.split(':', 1)[0]}:other"
        filtered_transitions[(src, tgt)] += count

    nodes = sorted({node for edge in filtered_transitions for node in edge})
    node_index = {name: idx for idx, name in enumerate(nodes)}
    sources = [node_index[src] for src, _ in filtered_transitions]
    targets = [node_index[tgt] for _, tgt in filtered_transitions]
    values = [int(count) for count in filtered_transitions.values()]

    outpath = outdir / f"sankey_trajectory_{scenario}_{model}.html"
    _plot_sankey(nodes, sources, targets, values, outpath, f"Tool Trajectories: {scenario} ({model})")


def plot_trajectory_lines(
    long_df: pd.DataFrame,
    outdir: Path,
    scenario: str,
    model: str,
    top_tools: int,
) -> None:
    if model == "ALL":
        sub = long_df[(long_df["scenario"] == scenario) & (long_df["model"] == "ALL")].copy()
    else:
        sub = long_df[(long_df["scenario"] == scenario) & (long_df["model"] == model)].copy()
    if sub.empty:
        print(f"[WARN] No trajectory line data for {scenario} ({model})")
        return

    grouped = (
        sub.groupby(["turn", "tool_name"])
        .size()
        .reset_index(name="count")
    )
    tool_totals = grouped.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = tool_totals.head(top_tools).index.tolist()
    grouped["tool_group"] = grouped["tool_name"].where(
        grouped["tool_name"].isin(keep_tools), other="other"
    )
    grouped = grouped.groupby(["turn", "tool_group"])["count"].sum().reset_index()
    totals = grouped.groupby("turn")["count"].sum().reset_index(name="total")
    grouped = grouped.merge(totals, on="turn", how="left")
    grouped["pct"] = grouped["count"] / grouped["total"]
    pivot = grouped.pivot(index="turn", columns="tool_group", values="pct").fillna(0.0)
    pivot_count = grouped.pivot(index="turn", columns="tool_group", values="count").fillna(0.0)
    totals_by_turn = totals.set_index("turn")["total"]

    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    x_vals = pivot.index.to_numpy()
    for idx, tool in enumerate(pivot.columns):
        color = LINE_PALETTE[idx % len(LINE_PALETTE)]
        y_vals = pivot[tool].to_numpy()
        k_vals = pivot_count[tool].to_numpy()
        n_vals = totals_by_turn.reindex(pivot.index).to_numpy()
        low, high = wilson_interval(k_vals, n_vals)
        ax.fill_between(x_vals, low, high, color=color, alpha=0.2, linewidth=0)
        ax.plot(
            x_vals,
            y_vals,
            marker="^" if str(tool) in TRIANGLE_ACTIONS else "o",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            color=color,
            label=tool,
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel("Turn")
    plt.ylabel("Tool Share")
    plt.title(f"Tool Share by Turn ({scenario}, {model})", fontsize=FONT_SCALE + 2, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="upper right", fontsize=FONT_SCALE - 2)
    outpath = outdir / f"line_trajectory_{scenario}_{model}.png"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {outpath}")


def plot_trajectory_heatmap(
    long_df: pd.DataFrame,
    outdir: Path,
    scenario: str,
    model: str,
    top_tools: int,
) -> None:
    if model == "ALL":
        sub = long_df[(long_df["scenario"] == scenario) & (long_df["model"] == "ALL")].copy()
    else:
        sub = long_df[(long_df["scenario"] == scenario) & (long_df["model"] == model)].copy()
    if sub.empty:
        print(f"[WARN] No trajectory heatmap data for {scenario} ({model})")
        return

    grouped = (
        sub.groupby(["turn", "tool_name"])
        .size()
        .reset_index(name="count")
    )
    tool_totals = grouped.groupby("tool_name")["count"].sum().sort_values(ascending=False)
    keep_tools = tool_totals.head(top_tools).index.tolist()
    grouped["tool_group"] = grouped["tool_name"].where(
        grouped["tool_name"].isin(keep_tools), other="other"
    )
    grouped = grouped.groupby(["turn", "tool_group"])["count"].sum().reset_index()
    pivot = grouped.pivot(index="turn", columns="tool_group", values="count").fillna(0.0)
    tool_order = _sort_tool_labels(pivot.columns.tolist())
    pivot = pivot[tool_order]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xlabel("Tool ID")
    ax.set_ylabel("Turn")
    ax.set_title(f"Tool Count Heatmap ({scenario}, {model})", fontsize=FONT_SCALE + 2, fontweight="bold")
    ax.set_xticks(range(len(tool_order)))
    ax.set_xticklabels(tool_order, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")

    outpath = outdir / f"heatmap_trajectory_{scenario}_{model}.png"
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze tool choice proportions and trajectories."
    )
    parser.add_argument(
        "--csv",
        default="logs/statistics/summary/turn_level_agent_actions.csv",
        help="Input CSV from extract_turn_reasoning_tools.py.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs_tool_usage",
        help="Output directory.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model name to plot (can repeat). Default: ALL.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Scenario name to plot trajectories (can repeat). Default: top scenarios.",
    )
    parser.add_argument("--top-tools", type=int, default=8, help="Top tools to keep.")
    parser.add_argument(
        "--top-scenarios", type=int, default=8, help="Top scenarios to keep."
    )
    parser.add_argument(
        "--include-none",
        action="store_true",
        help="Include empty tool choices as 'none'.",
    )
    parser.add_argument(
        "--tool-label",
        choices=["id", "name"],
        default="id",
        help="Use tool id or tool name as the label.",
    )
    parser.add_argument(
        "--tool-ids",
        default="1,2,3,4,5,6",
        help="Comma-separated tool ids to keep (only for --tool-label id).",
    )
    parser.add_argument(
        "--no-merge-scenario-variants",
        action="store_true",
        help="Do not merge scenario variants like *_1/_2.",
    )
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": FONT_SCALE,
        "axes.titlesize": FONT_SCALE + 2,
        "axes.labelsize": FONT_SCALE,
        "xtick.labelsize": FONT_SCALE - 1,
        "ytick.labelsize": FONT_SCALE - 1,
        "legend.fontsize": FONT_SCALE - 2,
    })

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df[~df["model"].astype(str).str.lower().str.contains("claude")]
    if not args.no_merge_scenario_variants:
        df["scenario"] = df["scenario"].apply(_normalize_scenario_name)
    all_models = sorted(df["model"].dropna().astype(str).unique().tolist())
    turn_cols = _extract_turn_columns(df.columns)
    if not turn_cols:
        raise SystemExit("No turn tool columns found.")

    allowed_ids = (
        {s.strip() for s in args.tool_ids.split(",") if s.strip()}
        if args.tool_label == "id"
        else set()
    )
    long_df = _build_long_df(
        df,
        turn_cols,
        include_none=args.include_none,
        label_mode=args.tool_label,
        allowed_ids=allowed_ids,
    )
    long_df = _add_all_models(long_df)
    tool_norm = long_df["tool_name"].astype(str).str.strip().str.lower()
    long_df = long_df[~tool_norm.isin({"", "none", "unknown"})].copy()

    scenario_share = compute_scenario_tool_share(long_df)
    scenario_csv = outdir / "tool_choice_by_model_scenario.csv"
    scenario_share.to_csv(scenario_csv, index=False)
    print(f"[OK] Wrote {scenario_csv}")

    turn_share = compute_turn_tool_share(long_df)
    turn_csv = outdir / "tool_choice_by_turn.csv"
    turn_share.to_csv(turn_csv, index=False)
    print(f"[OK] Wrote {turn_csv}")

    traj_summary = compute_trajectory_summary(
        df, turn_cols, args.tool_label, allowed_ids
    )
    traj_summary = _add_all_models(traj_summary)
    traj_csv = outdir / "trajectory_by_model_scenario.csv"
    traj_summary.to_csv(traj_csv, index=False)
    print(f"[OK] Wrote {traj_csv}")

    models = args.model or ["ALL"]
    for model in models:
        plot_turn_share(turn_share, outdir, model, args.top_tools)
        plot_turn_heatmap(turn_share, outdir, model, args.top_tools)
        plot_scenario_tool_lines(
            scenario_share, outdir, model, args.top_scenarios, args.top_tools
        )
        plot_scenario_tool_heatmap(
            scenario_share, outdir, model, args.top_scenarios, args.top_tools
        )

    if args.scenario:
        scenario_list = args.scenario
    else:
        scenario_list = sorted(
            scenario_share[scenario_share["model"] == "ALL"]["scenario"]
            .dropna()
            .astype(str)
            .unique()
            .tolist(),
            key=lambda s: s.lower(),
        )[: args.top_scenarios]

    for scenario in scenario_list:
        for model in models:
            plot_trajectory_lines(
                long_df,
                outdir,
                scenario,
                model,
                args.top_tools,
            )
            plot_trajectory_heatmap(
                long_df,
                outdir,
                scenario,
                model,
                args.top_tools,
            )


if __name__ == "__main__":
    main()
