"""Configuration loading for RAGWrapper.

Supports YAML, TOML, and JSON configuration files with environment variable expansion.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None
import json

# Load environment variables from .env file automatically
load_dotenv()


class Config:
    """Holds RAGWrapper configuration with sensible defaults."""

    def __init__(
        self,
        files: list[str] | str | None = None,
        db_path: str = "db",
        vector_store: dict[str, Any] | None = None,
        llm: dict[str, Any] | None = None,
        telegram: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        chunk_threshold: int = 10000,
        log_level: str = "INFO",
        context_file: str | None = None,
        character_file: str | None = None,
        allowed_user_ids: list[int] | None = None,
        allowed_chat_ids: list[int] | None = None,
        **_unused: Any,
    ):
        self.files = files if files is not None else "data"
        self.db_path = db_path
        self.vector_store = vector_store or {}
        self.llm = llm or {}
        self.telegram = telegram or {}
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_threshold = chunk_threshold
        self.log_level = log_level
        self.context_file = context_file
        self.character_file = character_file
        self.allowed_user_ids = allowed_user_ids or []
        self.allowed_chat_ids = allowed_chat_ids or []

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load configuration from a file (YAML, TOML, or JSON) with env var expansion."""
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
                raise ValueError(f"Unsupported config format: {path.suffix}")

        # Expand environment variables in the loaded data
        data = cls._expand_env_vars(data)
        
        return cls(**data)

    @staticmethod
    def _expand_env_vars(data: Any) -> Any:
        """Recursively expand environment variables in strings.
        
        Supports ${VAR} and ${VAR:default} syntax.
        """
        if isinstance(data, str):
            # Expand ${VAR} and ${VAR:default} patterns
            def replace_var(match):
                var_expr = match.group(1)
                if ':' in var_expr:
                    var_name, default = var_expr.split(':', 1)
                else:
                    var_name, default = var_expr, None
                return os.getenv(var_name, default) if default else os.getenv(var_name, '')
            return re.sub(r'\$\{([^}]+)\}', replace_var, data)
        elif isinstance(data, dict):
            return {k: Config._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [Config._expand_env_vars(item) for item in data]
        else:
            return data