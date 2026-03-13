"""Configuration loading for RAGWrapper.

Supports YAML, TOML, and JSON configuration files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    import toml
except ImportError:
    toml = None
import json


class Config:
    """Holds RAGWrapper configuration with sensible defaults."""

    def __init__(
        self,
        files: list[str] | str | None = None,
        db_path: str = "db",
        vector_store: dict[str, Any] | None = None,
        llm: dict[str, Any] | None = None,
        chunk_size: int = 1000,
        overlap: int = 200,
        log_level: str = "INFO",
        context_file: str | None = None,
        character_file: str | None = None,
        **_unused: Any,
    ):
        self.files = files if files is not None else "data"
        self.db_path = db_path
        self.vector_store = vector_store or {}
        self.llm = llm or {}
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.log_level = log_level
        self.context_file = context_file
        self.character_file = character_file

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load configuration from a file (YAML, TOML, or JSON)."""
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

        return cls(**data)
