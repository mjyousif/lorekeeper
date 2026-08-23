from abc import ABC, abstractmethod


class ImageGenerationProvider(ABC):
    """Abstract base class for image generation providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an image from a prompt and return a URL or file path/data."""
        pass
