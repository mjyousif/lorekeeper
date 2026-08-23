from __future__ import annotations

import logging
import os

from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles resolving paths, reading file contents, and detecting file changes."""

    def __init__(
        self, file_spec: list[str] | str, exclude_paths: list[str] | None = None
    ):
        """Initialize the document loader.

        Args:
            file_spec: A string or list of strings representing paths to files
                       or directories.
            exclude_paths: Optional list of file paths to exclude from loading/indexing.
        """
        self._file_spec = file_spec
        self.exclude_paths = (
            [os.path.abspath(p) for p in exclude_paths] if exclude_paths else []
        )
        logger.info(
            "DocumentLoader initializing with spec=%s, exclude=%d paths",
            file_spec,
            len(self.exclude_paths),
        )
        self.files = self.resolve_files(self._file_spec)
        self._manifest = self.scan_files()
        logger.info(
            "DocumentLoader ready: %d files, %d manifest entries",
            len(self.files),
            len(self._manifest),
        )

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
                            full_path = os.path.join(root, fname)
                            if os.path.abspath(full_path) not in self.exclude_paths:
                                results.append(full_path)
                            else:
                                logger.debug("Excluding file: %s", full_path)
            elif os.path.isfile(path):
                if os.path.abspath(path) not in self.exclude_paths:
                    results.append(path)
                else:
                    logger.debug("Excluding file: %s", path)
            else:
                logger.warning("Path does not exist or is not a file/dir: %s", path)

        logger.debug(
            "Resolved %d files from %d input paths", len(results), len(input_paths)
        )
        return results

    def scan_files(self) -> dict[str, tuple[float, int]]:
        """Scan the current set of data files and return a dict of path →
        (mtime, size).
        """
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
            added = set(current.keys()) - set(self._manifest.keys())
            removed = set(self._manifest.keys()) - set(current.keys())
            if added:
                logger.info("New files detected: %s", added)
            if removed:
                logger.info("Removed files detected: %s", removed)
            return True
        for path, info in current.items():
            if info != self._manifest.get(path):
                logger.info(
                    "File changed: %s (was %s, now %s)",
                    path,
                    self._manifest.get(path),
                    info,
                )
                return True
        logger.debug("No file changes detected")
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

        logger.debug("Reading file: %s", file_path)
        if file_path.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            logger.debug(
                "Read PDF %s: %d pages, %d chars",
                file_path,
                len(reader.pages),
                len(text),
            )
            return text
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug("Read text file %s: %d chars", file_path, len(content))
            return content
