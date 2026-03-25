from __future__ import annotations

import os
import logging
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles resolving paths, reading file contents, and detecting file changes."""

    def __init__(self, file_spec: list[str] | str):
        """Initialize the document loader.

        Args:
            file_spec: A string or list of strings representing paths to files or directories.
        """
        self._file_spec = file_spec
        self.files = self.resolve_files(self._file_spec)
        self._manifest = self.scan_files()

    def update_files(self, new_file_spec: list[str] | str | None = None) -> None:
        """Update the internal list of resolved files and manifest."""
        if new_file_spec is not None:
            self._file_spec = new_file_spec
        self.files = self.resolve_files(self._file_spec)
        self._manifest = self.scan_files()

    def resolve_files(self, input_paths: list[str] | str) -> list[str]:
        """Return a flat list of readable files.

        * If a directory is provided, walk it recursively and include any
          `.txt`, `.md` or `.pdf` files.
        * If a list is provided it may contain files or directories.
        * Nonexistent paths are skipped with a warning.

        Args:
            input_paths: The file or directory paths to resolve.

        Returns:
            A flat list of absolute or relative file paths.
        """
        allowed = (".txt", ".md", ".pdf")
        results: list[str] = []

        if isinstance(input_paths, str):
            input_paths = [input_paths]

        for path in input_paths:
            if os.path.isdir(path):
                for root, _, files in os.walk(path, followlinks=True):
                    for fname in files:
                        if fname.lower().endswith(allowed):
                            results.append(os.path.join(root, fname))
            elif os.path.isfile(path):
                results.append(path)
            else:
                logger.warning("Path does not exist or is not a file/dir: %s", path)

        return results

    def scan_files(self) -> dict[str, tuple[float, int]]:
        """Scan the current set of data files and return a dict of path → (mtime, size)."""
        manifest: dict[str, tuple[float, int]] = {}
        current_files = self.resolve_files(self._file_spec)
        for path in current_files:
            try:
                stat = os.stat(path)
                manifest[path] = (stat.st_mtime, stat.st_size)
            except Exception as e:
                logger.warning("Cannot stat %s: %s", path, e)
        return manifest

    def needs_rebuild(self) -> bool:
        """Compare current files against the stored manifest to detect changes."""
        current = self.scan_files()
        if set(current.keys()) != set(self._manifest.keys()):
            return True
        for path, info in current.items():
            if info != self._manifest.get(path):
                return True
        return False

    def read_file(self, file_path: str) -> str:
        """Read content from a file (supports .txt, .md, .pdf).

        Args:
            file_path: The path to the file to read.

        Returns:
            The text content of the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
