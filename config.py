from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or PROJECT_ROOT / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative_path: str, config: dict[str, Any] | None = None) -> Path:
    root = PROJECT_ROOT
    if config is None:
        config = load_config()
    return (root / relative_path).resolve()
