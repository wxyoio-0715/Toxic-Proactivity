"""Shared logging helpers and log directory resolution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from core.config import get_section
from core.config_utils import CONFIG_BASE_DIR_FIELD


def resolve_log_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    logging_cfg = get_section(config, "logging")
    base_dir = Path(config.get(CONFIG_BASE_DIR_FIELD, Path.cwd()))

    def _resolve(path_value: Any) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    root_dir = _resolve(logging_cfg.get("root_dir", "logs"))
    run_dir = _resolve(logging_cfg.get("simulation_run_dir", root_dir / "runs"))
    conversation_dir = _resolve(logging_cfg.get("conversation_dir", root_dir / "conversations"))
    stats_dir = _resolve(logging_cfg.get("statistics_dir", root_dir / "statistics"))
    stats_summary_dir = _resolve(
        logging_cfg.get("statistics_summary_dir", stats_dir / "summary")
    )
    stats_aggregate_dir = _resolve(
        logging_cfg.get("statistics_aggregate_dir", stats_dir / "aggregate")
    )
    workflow_dir = _resolve(logging_cfg.get("workflow_dir", root_dir / "workflow"))

    for path in (
        root_dir,
        run_dir,
        conversation_dir,
        stats_dir,
        stats_summary_dir,
        stats_aggregate_dir,
        workflow_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root_dir,
        "runs": run_dir,
        "conversations": conversation_dir,
        "statistics": stats_summary_dir,
        "statistics_summary": stats_summary_dir,
        "statistics_aggregate": stats_aggregate_dir,
        "workflow": workflow_dir,
    }


def setup_logger(
    name: str,
    *,
    log_file: Path,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
