from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles splitting text into manageable, overlapping chunks."""

    def __init__(
        self, chunk_size: int = 1000, overlap: int = 200, chunk_threshold: int = 10000
    ):
        """Initialize the text chunker.

        Args:
            chunk_size: The target size of each text chunk.
            overlap: The number of characters to overlap between chunks.
            chunk_threshold: If text is shorter than this, it is not chunked.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_threshold = chunk_threshold

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        If the text length is less than or equal to the chunk_threshold,
        it is returned as a single chunk.

        Args:
            text: The full text to be chunked.

        Returns:
            A list of text chunks.
        """
        if not text:
            return []

        if len(text) <= self.chunk_threshold:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks
