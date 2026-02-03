"""Smoke test for model connectivity using the configured profiles."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from misalignment_simulation.config_utils import CONFIG_BASE_DIR_FIELD, apply_profile, load_model_profiles
from core.config import load_config
from misalignment_simulation.models import ModelClient

LOGGER = logging.getLogger(__name__)

def run_smoke_test(config: Dict[str, Any], role: str, prompt: str) -> None:
    role_key = f"{role}_model"
    if role_key not in config:
        raise KeyError(f"Configuration missing section '{role_key}'")

    default_provider = (config.get("default_provider") or "anthropic").strip()
    default_api_key = (config.get("api_key") or "").strip()
    default_base_url = (config.get("base_url") or "").strip()

    base_dir = Path(config.get(CONFIG_BASE_DIR_FIELD, Path(__file__).resolve().parent))
    profiles = load_model_profiles(config.get("model_profiles_path"), relative_to=base_dir)

    model_cfg: Dict[str, Any] = apply_profile(config[role_key] or {}, profiles)

    provider = (model_cfg.get("provider") or default_provider).strip()
    model_name = (model_cfg.get("model_name") or "").strip()
    api_key = (model_cfg.get("api_key") or "").strip() or default_api_key
    base_url = (model_cfg.get("base_url") or default_base_url).strip()
    max_tokens = int(model_cfg.get("max_tokens", 1024))
    top_p = model_cfg.get("top_p")

    LOGGER.info("Preparing client for role=%s provider=%s model=%s", role, provider, model_name)
    client = ModelClient(
        provider,
        api_key=api_key,
        model_name=model_name,
        max_tokens=max_tokens,
        base_url=base_url,
        top_p=top_p if isinstance(top_p, (int, float)) else None,
    )

    LOGGER.info("Sending prompt (%d chars)", len(prompt))
    response = client.invoke(prompt)

    try:
        parsed = json.loads(response)
        LOGGER.info("Model returned valid JSON with keys: %s", list(parsed.keys()))
    except json.JSONDecodeError:
        LOGGER.info("Model returned non-JSON text response")

    print("=== Raw Model Response ===")
    print(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple LLM connectivity test")
    parser.add_argument(
        "--role",
        choices=["agent", "environment"],
        default="agent",
        help="Which model configuration block to use",
    )
    parser.add_argument(
        "--prompt",
        default="Please respond with an example of a short JSON object",
        help="Override the default test prompt",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config_path = Path(args.config).resolve() if args.config else None
    config = load_config(config_path)

    run_smoke_test(config, args.role, args.prompt)


if __name__ == "__main__":
    main()
