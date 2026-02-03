#!/usr/bin/env python3
"""Run all scenarios in scenario_generation/output for multiple agent
profiles and record the results."""

from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.config_utils import CONFIG_BASE_DIR_FIELD, apply_profile, load_model_profiles  # noqa: E402


DEFAULT_PROFILES = [
    "siliconflow-qwen3-32b",
    "siliconflow-deepseek-v3.2",
    "claude-haiku-4.5"
]

PROFILE_ALIASES = {
    "qwen/qwen3-32b": "qwen3-32b",
    "claude-haiku-4.5": "claude-haiku-4.5",
    "Qwen/Qwen3-32B": "siliconflow-qwen3-32b",
    "deepseek-ai/DeepSeek-V3.2": "siliconflow-deepseek-v3.2",
}


def _resolve_profile(name: str) -> str:
    key = name.strip()
    return PROFILE_ALIASES.get(key, key)


def _ensure_profile_exists(profile: str, profiles_path: Path) -> None:
    profiles = load_model_profiles(profiles_path, relative_to=profiles_path.parent.parent)
    if profile not in profiles:
        known = ", ".join(sorted(profiles.keys()))
        raise ValueError(f"Unknown model profile '{profile}'. Known profiles: {known}")


def _write_config(
    base_config: Dict,
    profile: str,
    runs: int | None,
    output_path: Path,
) -> None:
    config = copy.deepcopy(base_config)
    config.setdefault("agent_model", {})["profile"] = profile
    if runs is not None:
        config.setdefault("simulation", {})["runs_per_execution"] = runs
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return cleaned.strip("_") or "config"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all output scenarios for multiple agent profiles."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=DEFAULT_PROFILES,
        help="Model profile names (default: gemini-3-flash qwen3-32b claude-haiku-4.5).",
    )
    parser.add_argument(
        "--scenario-dir",
        default="scenario_generation/output",
        help="Scenario directory (default: scenario_generation/output).",
    )
    parser.add_argument(
        "--scenario-glob",
        default="*.yaml",
        help="Scenario glob pattern (default: *.yaml).",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Base config path (default: config/config.yaml).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override runs_per_execution (default: use config).",
    )
    parser.add_argument(
        "--keep-configs",
        action="store_true",
        help="Keep generated config files under config/.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Base config not found: {config_path}")

    scenario_dir = Path(args.scenario_dir).resolve()
    if not scenario_dir.exists():
        raise SystemExit(f"Scenario dir not found: {scenario_dir}")

    base_config = load_config(config_path)
    base_config.pop(CONFIG_BASE_DIR_FIELD, None)

    profiles_path = config_path.parent / "model_profiles.yaml"
    if not profiles_path.exists():
        raise SystemExit(f"Model profiles not found: {profiles_path}")

    python_exe = sys.executable

    for raw_profile in args.profiles:
        profile = _resolve_profile(raw_profile)
        _ensure_profile_exists(profile, profiles_path)

        config_name = _sanitize_filename(
            f"run_output_{profile}.yaml"
        )
        config_path_local = profiles_path.parent / config_name
        try:
            _write_config(base_config, profile, args.runs, config_path_local)
            cmd = [
                python_exe,
                "-m",
                "misalignment_simulation.main",
                "--scenario-dir",
                str(scenario_dir),
                "--scenario-glob",
                args.scenario_glob,
                "--config",
                str(config_path_local),
            ]
            print(f"Running profile {profile}...")
            subprocess.run(cmd, check=True)
        finally:
            if not args.keep_configs and config_path_local.exists():
                config_path_local.unlink()


if __name__ == "__main__":
    main()
