"""Compatibility shim for config utilities."""

from core.config_utils import (  # noqa: F401
    CONFIG_BASE_DIR_FIELD,
    apply_profile,
    infer_provider_from_model_name,
    load_model_profiles,
    resolve_path,
)
