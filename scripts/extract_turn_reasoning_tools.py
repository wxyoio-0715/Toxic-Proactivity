#!/usr/bin/env python3
"""Extract per-turn agent reasoning and tool choices from run logs
and write a CSV output."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


def iter_run_logs(base_dir: Path) -> Iterator[Tuple[str, str, Path]]:
    for model_dir in sorted(d for d in base_dir.iterdir() if d.is_dir()):
        model_name = model_dir.name
        for scenario_dir in sorted(d for d in model_dir.iterdir() if d.is_dir()):
            scenario_name = scenario_dir.name
            for run_path in sorted(scenario_dir.glob("run_*.json")):
                yield model_name, scenario_name, run_path


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _extract_agent_entry(turn: Dict[str, Any]) -> Dict[str, Any]:
    agent = turn.get("agent") or {}
    parsed = agent.get("parsed") or {}
    tool_choice = parsed.get("tool_choice") or {}
    return {
        "turn": turn.get("turn"),
        "reasoning": parsed.get("reasoning", ""),
        "tool_id": tool_choice.get("id"),
        "tool_name": tool_choice.get("name"),
        "tool_parameters": tool_choice.get("parameters"),
    }




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-turn agent reasoning and tool choices."
    )
    parser.add_argument(
        "--base-dir",
        default="experiment_results",
        help="Directory containing model/scenario run logs.",
    )
    parser.add_argument(
        "--output",
        default="logs/statistics/summary/turn_level_agent_actions.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    max_turn = 0
    for model_name, scenario_name, run_path in iter_run_logs(base_dir):
        with run_path.open("r", encoding="utf-8") as run_handle:
            data = json.load(run_handle)
        turn_map: Dict[int, Dict[str, Any]] = {}
        for turn in data.get("turns", []) or []:
            entry = _extract_agent_entry(turn)
            turn_id = entry.get("turn")
            if isinstance(turn_id, int):
                turn_map[turn_id] = entry
                if turn_id > max_turn:
                    max_turn = turn_id
        rows.append(
            {
                "model": model_name,
                "scenario": scenario_name,
                "run_file": str(run_path),
                "turn_map": turn_map,
            }
        )

    header = ["model", "scenario", "run_file"]
    for idx in range(1, max_turn + 1):
        header.extend(
            [
                f"turn{idx}_reasoning",
                f"turn{idx}_tool_id",
                f"turn{idx}_tool_name",
                f"turn{idx}_tool_params",
            ]
        )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            turn_map = row["turn_map"]
            row_values = [
                row["model"],
                row["scenario"],
                row["run_file"],
            ]
            for idx in range(1, max_turn + 1):
                entry = turn_map.get(idx, {})
                row_values.extend(
                    [
                        _stringify(entry.get("reasoning", "")),
                        _stringify(entry.get("tool_id", "")),
                        _stringify(entry.get("tool_name", "")),
                        _stringify(entry.get("tool_parameters", "")),
                    ]
                )
            writer.writerow(row_values)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
