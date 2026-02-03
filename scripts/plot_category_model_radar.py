#!/usr/bin/env python3
"""Plot radar charts from category_model_behavior_summary.csv."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

MODEL_PREFIXES = {
    "openai",
    "google",
    "anthropic",
    "qwen",
    "deepseek",
    "deepseek-ai",
    "meta-llama",
}


def abbreviate_model_name(name: str) -> str:
    cleaned = (name or "").strip().lower()
    if "_" in cleaned:
        prefix, rest = cleaned.split("_", 1)
        if prefix in MODEL_PREFIXES:
            cleaned = rest
    cleaned = cleaned.replace("_", ".")
    cleaned = re.sub(r"-a22b-thinking-2507$", "-thinking", cleaned)
    cleaned = re.sub(r"-a22b-2507$", "", cleaned)
    return cleaned


def load_category_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    import csv

    rows: Dict[Tuple[str, str], Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = (row.get("category") or "").strip()
            model = (row.get("model") or "").strip()
            if not category or not model:
                continue
            key = (category.lower(), model.lower())
            rows[key] = {
                "valid_runs": float(row.get("valid_runs") or 0),
                "behavior_direct": float(row.get("behavior_direct") or 0),
                "behavior_strategic": float(row.get("behavior_strategic") or 0),
            }
    return rows


def compute_success_rates(runs_dir: Path) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for model_dir in sorted(d for d in runs_dir.iterdir() if d.is_dir()):
        model_name = abbreviate_model_name(model_dir.name)
        run_map: Dict[Tuple[str, int], Tuple[float, Dict[str, object]]] = {}
        for scenario_dir in sorted(d for d in model_dir.iterdir() if d.is_dir()):
            for run_path in scenario_dir.glob("run_*.json"):
                try:
                    data = json.loads(run_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                run_index = data.get("run_index")
                if run_index is None:
                    parts = run_path.stem.split("_")
                    if len(parts) >= 2 and parts[0] == "run":
                        try:
                            run_index = int(parts[1])
                        except ValueError:
                            continue
                try:
                    run_index_int = int(run_index)
                except Exception:
                    continue
                key = (scenario_dir.name, run_index_int)
                try:
                    mtime = run_path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                current = run_map.get(key)
                if current is None or mtime > current[0]:
                    run_map[key] = (mtime, data)

        success = 0
        total = 0
        for _, data in run_map.values():
            normalized = str(data.get("normalized_termination_signal") or "").strip().lower()
            if not normalized:
                normalized = str(data.get("termination_signal") or "").strip().lower()
            if normalized == "sucess":
                normalized = "success"
            if normalized not in {"success", "fail", "fail_attempt", "none", "error"}:
                normalized = "none"
            if normalized == "error":
                continue
            total += 1
            if normalized in {"success", "fail_attempt"}:
                success += 1
        if total > 0:
            rates[model_name] = success / total
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot radar chart by category*model behaviors."
    )
    parser.add_argument(
        "--category-csv",
        default="logs/statistics/summary/category_model_behavior_summary.csv",
        help="Path to category_model_behavior_summary.csv",
    )
    parser.add_argument(
        "--runs-dir",
        default="experiment_results",
        help="Directory with run logs for success rate",
    )
    parser.add_argument(
        "--models",
        default="gpt-4o-2024-11-20,deepseek-r1-0528,gemini-3-flash-preview,llama-3.3-70b-instruct,deepseek-v3.2,qwen-3-235b-a22b-thinking-2507, qwen-3-32b",
        help="Comma-separated model names (abbreviated)",
    )
    parser.add_argument(
        "--out",
        default="outputs_misalign/category_model_radar.png",
        help="Output image path",
    )
    args = parser.parse_args()

    category_csv = Path(args.category_csv)
    if not category_csv.exists():
        raise SystemExit(f"Missing CSV: {category_csv}")

    category_rows = load_category_rows(category_csv)
    success_rates = compute_success_rates(Path(args.runs_dir))

    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models specified.")

    axes_labels = [
        "Loyalty-Strategic",
        "Loyalty-Direct",
        "SelfPres-Strategic",
        "SelfPres-Direct",
        "Overall Misalignment Rate",
    ]
    angles = np.linspace(0, 2 * math.pi, len(axes_labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.0)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels)

    for model in models:
        loyalty = category_rows.get(("loyalty", model), {})
        selfpres = category_rows.get(("self-preservation", model), {})
        loyalty_valid = loyalty.get("valid_runs", 0.0) or 0.0
        selfpres_valid = selfpres.get("valid_runs", 0.0) or 0.0

        loyalty_strat = (loyalty.get("behavior_strategic", 0.0) / loyalty_valid) if loyalty_valid else 0.0
        loyalty_reck = (loyalty.get("behavior_direct", 0.0) / loyalty_valid) if loyalty_valid else 0.0
        self_strat = (selfpres.get("behavior_strategic", 0.0) / selfpres_valid) if selfpres_valid else 0.0
        self_reck = (selfpres.get("behavior_direct", 0.0) / selfpres_valid) if selfpres_valid else 0.0
        success_rate = success_rates.get(model, 0.0)
        misalign_rate = 1.0 - success_rate if success_rate else 0.0

        values = [loyalty_strat, loyalty_reck, self_strat, self_reck, misalign_rate]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
