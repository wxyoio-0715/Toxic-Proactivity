#!/usr/bin/env python3
"""Aggregate behavior distributions by category × model from
all_runs_summary.csv and write a CSV summary."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def _to_int(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate behavior distribution by category and model."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to all_runs_summary.csv",
    )
    parser.add_argument(
        "--out",
        default="logs/statistics/summary/category_model_behavior_summary.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise SystemExit("Input CSV has no rows.")

    behavior_cols = [col for col in rows[0].keys() if col.startswith("behavior_")]
    if not behavior_cols:
        raise SystemExit("Input CSV does not contain behavior_* columns.")

    totals: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {
        "valid_runs": 0,
        "avg_turn_sum": 0,
        **{col: 0 for col in behavior_cols},
    })

    for row in rows:
        category = row.get("category", "").strip()
        model = row.get("model", "").strip()
        if not category or not model:
            continue
        key = (category, model)
        totals[key]["valid_runs"] += _to_int(row.get("valid_runs"))
        try:
            avg_turn = float(row.get("avg_turn", 0) or 0)
        except (TypeError, ValueError):
            avg_turn = 0.0
        totals[key]["avg_turn_sum"] += avg_turn * _to_int(row.get("valid_runs"))
        for col in behavior_cols:
            totals[key][col] += _to_int(row.get(col))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["category", "model", "valid_runs", "avg_turn"]
    header.extend(behavior_cols)
    header.extend([f"{col}_pct" for col in behavior_cols])

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for (category, model) in sorted(totals.keys()):
            entry = totals[(category, model)]
            valid_runs = entry["valid_runs"]
            avg_turn = round((entry["avg_turn_sum"] / valid_runs) if valid_runs else 0.0, 2)
            counts = [entry[col] for col in behavior_cols]
            pcts = [
                round((entry[col] / valid_runs * 100) if valid_runs else 0.0, 2)
                for col in behavior_cols
            ]
            writer.writerow([category, model, valid_runs, avg_turn, *counts, *pcts])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
