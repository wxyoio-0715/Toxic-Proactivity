#!/usr/bin/env python3
"""Summarize statistics JSON files into tabular reports.

Usage:
    python summarize_statistics.py [--logs-dir LOGS_DIR] [--output OUTPUT_FORMAT]

Output formats:
    - table: console table (default)
    - csv: CSV file
    - markdown: Markdown table
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas is not installed; falling back to a simple table output.")

MODEL_ALIASES = {
    "qwen_qwen3-32b": "qwen_qwen3-32b",
}


def normalize_model_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        return cleaned
    alias = MODEL_ALIASES.get(cleaned.lower())
    return alias or cleaned


def parse_statistics_filename(filename: str) -> tuple[str, str] | None:
    """Parse scenario and model names from a statistics filename.

    Format: {scenario}_{model}_{timestamp}.json

    Since scenario and model can include underscores, we match the timestamp
    suffix and then split on the last underscore.

    Returns:
        (scenario_name, model_name) or None if the pattern does not match.
    """
    # Remove file extension.
    stem = Path(filename).stem
    
    # Match timestamp suffix: YYYYMMDDTHHMMSSZ or YYYYMMDDTHHMMSS...Z.
    # Pattern: at least 8 digits + T + at least 6 digits + Z.
    timestamp_pattern = r'_(\d{8}T\d{6,}Z)$'
    timestamp_match = re.search(timestamp_pattern, stem)
    
    if timestamp_match:
        # Prefix before timestamp contains scenario and model.
        prefix = stem[:timestamp_match.start()]
        # Split scenario and model on the last underscore.
        last_underscore = prefix.rfind('_')
        if last_underscore > 0:
            scenario_name = prefix[:last_underscore]
            model_name = prefix[last_underscore + 1:]
            return scenario_name, model_name
    
    return None


def extract_environment_model(conversation_log_path: Path, logs_dir: Path) -> Optional[str]:
    """Extract the environment model name from a conversation log.

    Args:
        conversation_log_path: Relative or absolute path to the conversation log.
        logs_dir: Logs directory.

    Returns:
        Environment model name, or None if unavailable.
    """
    # Resolve relative paths.
    if not conversation_log_path.is_absolute():
        full_path = logs_dir / conversation_log_path
    else:
        full_path = conversation_log_path
    
    if not full_path.exists():
        return None
    
    try:
        with full_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Attempt to extract environment model info from history.
        # This requires model metadata in the log; otherwise return None.
        # The current logs may not include it, so keep None for now.
        return None
    except Exception:
        return None


def load_statistics_file(file_path: Path, logs_dir: Path) -> Optional[Dict[str, Any]]:
    """Load a statistics JSON file and extract relevant fields.

    Returns:
        A dictionary of extracted statistics, or None if parsing fails.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Extract scenario and model from filename.
        parsed = parse_statistics_filename(file_path.name)
        if not parsed:
            print(f"Warning: unable to parse filename format: {file_path.name}")
            return None
        
        scenario_name, agent_model = parsed
        agent_model = normalize_model_name(agent_model)
        
        # Extract statistics fields.
        summary = data.get("summary", {})
        turn_stats = data.get("turn_statistics", {})
        error_summary = summary.get("error", {}) if isinstance(summary, dict) else {}
        terminal_tools = summary.get("terminal_tools", {})
        malign_terminal = terminal_tools.get("malign_terminal", {}) if isinstance(terminal_tools, dict) else {}
        benign_terminal = terminal_tools.get("benign_terminal", {}) if isinstance(terminal_tools, dict) else {}
        behavior_summary = summary.get("behavior_summary", {}) if isinstance(summary, dict) else {}
        behavior_classes = behavior_summary.get("classes", {}) if isinstance(behavior_summary, dict) else {}
        ttf_summary = summary.get("ttf_summary", {}) if isinstance(summary, dict) else {}

        def _behavior_metric(key: str, field: str, default_value: Any) -> Any:
            if not isinstance(behavior_classes, dict):
                return default_value
            return behavior_classes.get(key, {}).get(field, default_value)
        
        # Attempt to get environment model from the first detailed result (if present).
        env_model = "unknown"  # Default value.
        detailed_results = data.get("detailed_results", [])
        if detailed_results:
            # Try to extract from conversation logs, but keep unknown for now.
            # If needed, load from config.yaml instead.
            pass
        
        return {
            "scenario": scenario_name,
            "agent_model": agent_model,
            "env_model": env_model,
            "total_runs": data.get("total_runs", 0),
            "misalignment_severe": summary.get("misalignment_severe", {}).get("count", 0),
            "misalignment_mild": summary.get("misalignment_mild", {}).get("count", 0),
            "misalignment_total": summary.get("misalignment_total", {}).get("count", 0),
            "success": summary.get("success", {}).get("count", 0),
            "none": summary.get("none", {}).get("count", 0),
            "error": summary.get("error", {}).get("count", 0),
            "misalignment_rate": summary.get("misalignment_total", {}).get("percentage", 0.0),
            "average_turns": turn_stats.get("average_turns", 0.0),
            "timestamp": data.get("timestamp_utc", ""),
            "file_path": str(file_path),
            "terminal_malign_count": malign_terminal.get("count", 0),
            "terminal_malign_rate": malign_terminal.get("percentage", 0.0),
            "terminal_benign_count": benign_terminal.get("count", 0),
            "terminal_benign_rate": benign_terminal.get("percentage", 0.0),
            "success_count": summary.get("success", {}).get("count", 0),
            "success_rate": summary.get("success", {}).get("percentage", 0.0),
            "fail_count": summary.get("fail", {}).get("count", 0),
            "fail_rate": summary.get("fail", {}).get("percentage", 0.0),
            "fail_attempt_count": summary.get("fail_attempt", {}).get("count", 0),
            "fail_attempt_rate": summary.get("fail_attempt", {}).get("percentage", 0.0),
            "none_count": summary.get("none", {}).get("count", 0),
            "none_rate": summary.get("none", {}).get("percentage", 0.0),
            "error_count": error_summary.get("count", 0),
            "error_rate": error_summary.get("percentage", 0.0),
            "strategic_misalignment_count": _behavior_metric("strategic_misalignment", "count", 0),
            "strategic_misalignment_rate": _behavior_metric("strategic_misalignment", "percentage", 0.0),
            "reckless_misalignment_count": _behavior_metric("reckless_misalignment", "count", 0),
            "reckless_misalignment_rate": _behavior_metric("reckless_misalignment", "percentage", 0.0),
            "failed_malignancy_count": _behavior_metric("failed_malignancy", "count", 0),
            "failed_malignancy_rate": _behavior_metric("failed_malignancy", "percentage", 0.0),
            "robust_alignment_count": _behavior_metric("robust_alignment", "count", 0),
            "robust_alignment_rate": _behavior_metric("robust_alignment", "percentage", 0.0),
            "ttf_count": ttf_summary.get("count", 0),
            "ttf_average": ttf_summary.get("average"),
            "ttf_min": ttf_summary.get("min"),
            "ttf_max": ttf_summary.get("max"),
        }
    except Exception as e:
        print(f"Error: failed to load file {file_path}: {e}")
        return None


def collect_statistics(logs_dir: Path, config_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Collect data from all statistics files.

    Args:
        logs_dir: Logs directory.
        config_path: Config path (used to load environment model info).

    Returns:
        List of statistics dictionaries.
    """
    statistics: List[Dict[str, Any]] = []
    
    # Load environment model configuration.
    env_model_name = "unknown"
    if config_path and config_path.exists():
        try:
            from misalignment_simulation.config_utils import apply_profile, load_model_profiles, CONFIG_BASE_DIR_FIELD
            from core.config import load_config

            config = load_config(config_path)
            config_base_dir = Path(config.get(CONFIG_BASE_DIR_FIELD, config_path.parent))
            model_profiles = load_model_profiles(
                config.get("model_profiles_path"),
                relative_to=config_base_dir,
            )
            env_model_cfg = apply_profile(config.get("environment_model") or {}, model_profiles)
            env_model_name = env_model_cfg.get("model_name", "unknown").strip()
        except Exception as e:
            print(f"Warning: failed to read environment model from config: {e}")
    
    # Scan all statistics JSON files.
    for file_path in logs_dir.glob("*_*_*.json"):
        # Skip conversation logs (their filenames are longer).
        if file_path.name.startswith("conversation_"):
            continue
        
        stats = load_statistics_file(file_path, logs_dir)
        if stats:
            # Set environment model.
            stats["env_model"] = env_model_name
            statistics.append(stats)
    
    return statistics


def print_table(statistics: List[Dict[str, Any]]) -> None:
    """Print a simple text table."""
    if not statistics:
        print("No statistics files found.")
        return
    
    # Define columns.
    columns = [
        ("Scenario", "scenario", 25),
        ("Agent Model", "agent_model", 15),
        ("Environment Model", "env_model", 18),
        ("Runs", "total_runs", 6),
        ("SUCCESS(%)", "success_rate", 10),
        ("FAIL(%)", "fail_rate", 8),
        ("FAIL_ATTEMPT(%)", "fail_attempt_rate", 14),
        ("FAIL_ATTEMPT", "fail_attempt_count", 13),
        ("NONE(%)", "none_rate", 8),
        ("ERROR(%)", "error_rate", 8),
        ("Strategic Misalign(%)", "strategic_misalignment_rate", 18),
        ("Reckless Misalign(%)", "reckless_misalignment_rate", 18),
        ("Failed Malignancy(%)", "failed_malignancy_rate", 18),
        ("Robust Alignment(%)", "robust_alignment_rate", 18),
        ("TTF Avg", "ttf_average", 8),
        ("Terminal Malign(%)", "terminal_malign_rate", 18),
        ("Terminal Malign", "terminal_malign_count", 15),
        ("Terminal Benign(%)", "terminal_benign_rate", 18),
        ("Terminal Benign", "terminal_benign_count", 15),
        ("Avg Turns", "average_turns", 10),
    ]
    
    # Print header row.
    header = " | ".join(f"{col[0]:<{col[2]}}" for col in columns)
    print(header)
    print("-" * len(header))
    
    # Print data rows.
    for stats in sorted(statistics, key=lambda x: (x["scenario"], x["agent_model"])):
        row = []
        for _, key, width in columns:
            value = stats.get(key, "")
            if isinstance(value, float):
                if key in {
                    "misalignment_rate",
                    "terminal_malign_rate",
                    "terminal_benign_rate",
                    "error_rate",
                    "success_rate",
                    "fail_rate",
                    "fail_attempt_rate",
                    "none_rate",
                    "strategic_misalignment_rate",
                    "reckless_misalignment_rate",
                    "failed_malignancy_rate",
                    "robust_alignment_rate",
                }:
                    row.append(f"{value:.2f}%".rjust(width))
                else:
                    row.append(f"{value:.2f}".rjust(width))
            else:
                row.append(str(value)[:width].ljust(width))
        print(" | ".join(row))


def export_csv(statistics: List[Dict[str, Any]], output_path: Path) -> None:
    """Export results to a CSV file."""
    if HAS_PANDAS:
        df = pd.DataFrame(statistics)
        # Reorder columns.
        columns_order = [
            "scenario", "agent_model", "env_model", "total_runs",
            "success_count", "success_rate",
            "fail_count", "fail_rate",
            "fail_attempt_count", "fail_attempt_rate",
            "none_count", "none_rate",
            "error_count", "error_rate",
            "strategic_misalignment_count", "strategic_misalignment_rate",
            "reckless_misalignment_count", "reckless_misalignment_rate",
            "failed_malignancy_count", "failed_malignancy_rate",
            "robust_alignment_count", "robust_alignment_rate",
            "ttf_count", "ttf_average", "ttf_min", "ttf_max",
            "terminal_malign_rate", "terminal_malign_count",
            "terminal_benign_rate", "terminal_benign_count",
            "average_turns", "timestamp"
        ]
        # Keep only existing columns.
        columns_order = [col for col in columns_order if col in df.columns]
        df = df[columns_order]
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"CSV saved to: {output_path}")
    else:
        # Simple CSV fallback.
        if not statistics:
            return
        
        columns = [
            "scenario", "agent_model", "env_model", "total_runs",
            "success_count", "success_rate",
            "fail_count", "fail_rate",
            "fail_attempt_count", "fail_attempt_rate",
            "none_count", "none_rate",
            "error_count", "error_rate",
            "strategic_misalignment_count", "strategic_misalignment_rate",
            "reckless_misalignment_count", "reckless_misalignment_rate",
            "failed_malignancy_count", "failed_malignancy_rate",
            "robust_alignment_count", "robust_alignment_rate",
            "ttf_count", "ttf_average", "ttf_min", "ttf_max",
            "terminal_malign_rate", "terminal_malign_count",
            "terminal_benign_rate", "terminal_benign_count",
            "average_turns", "timestamp"
        ]
        
        with output_path.open("w", encoding="utf-8") as f:
            # Write header.
            f.write(",".join(columns) + "\n")
            
            # Write data rows.
            for stats in sorted(statistics, key=lambda x: (x["scenario"], x["agent_model"])):
                row = []
                for col in columns:
                    value = stats.get(col, "")
                    if isinstance(value, str) and "," in value:
                        value = f'"{value}"'
                    row.append(str(value))
                f.write(",".join(row) + "\n")
        
        print(f"CSV saved to: {output_path}")


def export_markdown(statistics: List[Dict[str, Any]], output_path: Path) -> None:
    """Export results to a Markdown table."""
    if not statistics:
        return
    
    columns = [
        ("Scenario", "scenario"),
        ("Agent Model", "agent_model"),
        ("Environment Model", "env_model"),
        ("Runs", "total_runs"),
        ("SUCCESS(%)", "success_rate"),
        ("FAIL(%)", "fail_rate"),
        ("FAIL_ATTEMPT(%)", "fail_attempt_rate"),
        ("FAIL_ATTEMPT", "fail_attempt_count"),
        ("NONE(%)", "none_rate"),
        ("ERROR(%)", "error_rate"),
        ("Strategic Misalign(%)", "strategic_misalignment_rate"),
        ("Reckless Misalign(%)", "reckless_misalignment_rate"),
        ("Failed Malignancy(%)", "failed_malignancy_rate"),
        ("Robust Alignment(%)", "robust_alignment_rate"),
        ("TTF Avg", "ttf_average"),
        ("Terminal Malign(%)", "terminal_malign_rate"),
        ("Terminal Malign", "terminal_malign_count"),
        ("Terminal Benign(%)", "terminal_benign_rate"),
        ("Terminal Benign", "terminal_benign_count"),
        ("Avg Turns", "average_turns"),
    ]
    
    with output_path.open("w", encoding="utf-8") as f:
        # Write header.
        header = "| " + " | ".join(col[0] for col in columns) + " |"
        f.write(header + "\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        
        # Write data rows.
        for stats in sorted(statistics, key=lambda x: (x["scenario"], x["agent_model"])):
            row = []
            for _, key in columns:
                value = stats.get(key, "")
                if isinstance(value, float):
                    if key in {
                        "misalignment_rate",
                        "terminal_malign_rate",
                        "terminal_benign_rate",
                        "error_rate",
                        "success_rate",
                        "fail_rate",
                        "fail_attempt_rate",
                        "none_rate",
                        "strategic_misalignment_rate",
                        "reckless_misalignment_rate",
                        "failed_malignancy_rate",
                        "robust_alignment_rate",
                    }:
                        row.append(f"{value:.2f}%")
                    else:
                        row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            f.write("| " + " | ".join(row) + " |\n")
    
    print(f"Markdown saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize statistics JSON files and generate tabular reports"
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Logs directory path (default: logs folder under current directory)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config path (used to read environment model info; default: config/config.yaml)"
    )
    parser.add_argument(
        "--output",
        choices=["table", "csv", "markdown", "all"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output file path (used for csv and markdown)"
    )
    
    args = parser.parse_args()

    run_summary(
        logs_dir=args.logs_dir,
        config_path=args.config,
        output=args.output,
        output_file=args.output_file,
    )


def run_summary(
    *,
    logs_dir: Optional[Path],
    config_path: Optional[Path],
    output: str,
    output_file: Optional[Path],
) -> None:
    if logs_dir:
        resolved_logs_dir = logs_dir
    else:
        try:
            from core.config import load_config
            from core.logging_utils import resolve_log_dirs

            config = load_config(config_path)
            resolved_logs_dir = resolve_log_dirs(config)["statistics_summary"]
        except Exception:
            script_dir = Path(__file__).resolve().parent
            resolved_logs_dir = script_dir / "logs"

    if not resolved_logs_dir.exists():
        print(f"Error: logs directory does not exist: {resolved_logs_dir}")
        return

    resolved_config_path = config_path or (Path(__file__).resolve().parents[1] / "config" / "config.yaml")

    print(f"Scanning logs directory: {resolved_logs_dir}")
    statistics = collect_statistics(resolved_logs_dir, resolved_config_path)

    if not statistics:
        print("No statistics files found.")
        return

    print(f"Found {len(statistics)} statistics files.\n")

    if output in ("table", "all"):
        print_table(statistics)
        print()

    if output in ("csv", "all"):
        csv_path = output_file or (resolved_logs_dir / "statistics_summary.csv")
        export_csv(statistics, csv_path)

    if output in ("markdown", "all"):
        md_path = output_file or (resolved_logs_dir / "statistics_summary.md")
        export_markdown(statistics, md_path)


if __name__ == "__main__":
    main()
