"""Helper utilities for loading shared configuration artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_BASE_DIR_FIELD = "_config_base_dir"


def infer_provider_from_model_name(model_name: str) -> str:
    """Infer provider from model name prefixes."""
    if not model_name:
        raise ValueError("Model name cannot be empty")

    model_name_lower = model_name.lower().strip()

    if any(model_name_lower.startswith(prefix) for prefix in ["gpt-", "o1-", "o3-"]):
        return "openai"

    if any(model_name_lower.startswith(prefix) for prefix in ["claude-", "sonnet-", "haiku-", "opus-"]):
        return "anthropic"

    if any(model_name_lower.startswith(prefix) for prefix in ["gemini-", "gemma-"]):
        return "google"

    if any(model_name_lower.startswith(prefix) for prefix in ["deepseek-", "deepseek"]):
        return "deepseek"

    raise ValueError(
        f"Cannot infer provider from model name '{model_name}'. "
        "Supported model prefixes: gpt-*, o1-*, o3-* (OpenAI), "
        "claude-*, sonnet-*, haiku-*, opus-* (Anthropic), "
        "gemini-*, gemma-* (Google), deepseek-* (DeepSeek)"
    )


def resolve_path(path_value: Any, *, relative_to: Path | None = None) -> Path:
    if path_value is None:
        raise ValueError("Path value is not defined")

    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        base_dir = relative_to or Path.cwd()
        path = (base_dir / path).resolve()
    return path


def load_model_profiles(path_value: Any, *, relative_to: Path | None = None) -> Dict[str, Dict[str, Any]]:
    if not path_value:
        return {}

    path = resolve_path(path_value, relative_to=relative_to)
    if not path.exists():
        raise FileNotFoundError(f"Model profiles file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    local_path = path.with_name(f"{path.stem}.local{path.suffix}")
    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as handle:
            local_raw = yaml.safe_load(handle) or {}
        if isinstance(local_raw, dict):
            for name, entry in local_raw.items():
                if not isinstance(entry, dict):
                    raise ValueError(f"Profile '{name}' in {local_path} must be a mapping of settings")
                base_entry = raw.get(name) if isinstance(raw, dict) else None
                if isinstance(base_entry, dict):
                    merged = base_entry.copy()
                    merged.update(entry)
                    raw[name] = merged
                else:
                    raw[name] = entry.copy()

    if not isinstance(raw, dict):
        raise ValueError(f"Model profiles file must contain a mapping at the top level: {path}")

    profiles: Dict[str, Dict[str, Any]] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Profile '{name}' must be a mapping of settings")
        profiles[name] = entry.copy()
    return profiles


def apply_profile(model_cfg: Dict[str, Any], profiles: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply model profile and infer provider if needed."""
    profile_name = str(model_cfg.get("profile", "")).strip()
    model_name = model_cfg.get("model_name", "").strip()

    if profile_name:
        if profile_name not in profiles:
            raise KeyError(f"Profile '{profile_name}' was not found in the loaded model profiles")
        merged = profiles[profile_name].copy()
        merged.update(model_cfg)
        if not merged.get("model_name"):
            merged["model_name"] = profile_name
    elif model_name:
        merged = dict(model_cfg)
        merged["model_name"] = model_name
    else:
        merged = dict(model_cfg)

    if not merged.get("provider"):
        final_model_name = merged.get("model_name", "").strip()
        if final_model_name:
            inferred_provider = infer_provider_from_model_name(final_model_name)
            merged["provider"] = inferred_provider

    return merged
