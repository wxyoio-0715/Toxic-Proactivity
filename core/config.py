"""Load application configuration from a single YAML file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from core.config_utils import CONFIG_BASE_DIR_FIELD, resolve_path

CONFIG_ENV_VAR = "MISALIGNMENT_CONFIG_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    config_path = resolve_path(
        path or os.getenv(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH,
        relative_to=Path.cwd(),
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    local_path = config_path.with_name(f"{config_path.stem}.local{config_path.suffix}")
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as handle:
            local_config = yaml.safe_load(handle) or {}
        if isinstance(local_config, dict):
            config = _merge_dict(config, local_config)
        else:
            raise ValueError(f"Local config must be a mapping at the top level: {local_path}")

    if not config.get("api_key"):
        env_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        config["api_key"] = env_key
    if not config.get("api_key") and config.get("api_key_wxy"):
        config["api_key"] = str(config.get("api_key_wxy") or "").strip()

    base_dir = config_path.parent
    if base_dir.name == "config":
        base_dir = base_dir.parent
    config[CONFIG_BASE_DIR_FIELD] = str(base_dir)
    return config


def get_section(config: Dict[str, Any], key: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    section = config.get(key)
    if isinstance(section, dict):
        return section
    return default or {}
