from abc import ABC, abstractmethod


class MusicGenerationProvider(ABC):
    """Abstract base class for music generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate music from a prompt and return a URL or markdown."""
        pass
