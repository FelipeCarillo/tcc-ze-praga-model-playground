"""Carregamento e merge de configurações YAML."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carrega um arquivo YAML e retorna como dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_model_config(
    model_config_path: str | Path,
    base_config_path: str | Path = "configs/base.yaml",
) -> dict[str, Any]:
    """Carrega base.yaml e aplica overrides do config do modelo."""
    base = load_config(base_config_path)
    override = load_config(model_config_path)
    return _deep_merge(base, override)
