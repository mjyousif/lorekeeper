from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None

import json

load_dotenv()


class Config(BaseSettings):
    """Typed RAGWrapper config with validation."""

    files: str | list[str] = "data"
    db_path: str = "db"
    vector_store: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    telegram: dict[str, Any] | None = None
    chunk_size: int = 1000
    overlap: int = 200
    chunk_threshold: int = 10000
    log_level: str = "INFO"
    context_file: str | None = None
    character_file: str | None = None
    allowed_user_ids: list[int] | None = None
    allowed_chat_ids: list[int] | None = None

    model_config = SettingsConfigDict(env_ignore_empty=True, extra="ignore")

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load from YAML/TOML/JSON with env expansion."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in {".yaml", ".yml"}:
                if yaml is None:
                    raise ImportError("PyYAML is required to load YAML files")
                data = yaml.safe_load(f) or {}
            elif path.suffix == ".toml":
                if toml is None:
                    raise ImportError("toml is required to load TOML files")
                data = toml.load(f)
            elif path.suffix == ".json":
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported format: {path.suffix}")

        data = _expand_env_vars(data)  # Module-level function, no cls issue
        return cls(**data)


def _expand_env_vars(data: Any) -> Any:
    """Recursively expand ${VAR} and ${VAR:default} in strings."""
    if isinstance(data, str):
        def replace_var(match):
            var_expr = match.group(1)
            if ":" in var_expr:
                var_name, default = var_expr.split(":", 1)
            else:
                var_name, default = var_expr, None
            return os.getenv(var_name, default) if default is not None else os.getenv(var_name, "")
        return re.sub(r"\$\{([^}]+)\}", replace_var, data)
    elif isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    return data


@lru_cache()
def get_config(config_path: str = "config.yaml") -> Config:
    """Cached config factory. Loads from file, falls back to defaults."""
    try:
        return Config.from_file(config_path)
    except FileNotFoundError:
        return Config()
