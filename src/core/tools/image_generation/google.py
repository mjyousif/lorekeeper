import base64
import logging
from typing import Any, Dict

from .base import ImageGenerationProvider

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore

logger = logging.getLogger(__name__)


class GoogleImageGenerationProvider(ImageGenerationProvider):
    """Image generation using Google GenAI (Imagen 3)."""

    def __init__(self, config: Dict[str, Any]):
        if genai is None:
            raise ImportError(
                "google-genai package is required for Google Image Generation"
            )

        self.api_key = config.get("api_key")
        self.model = config.get("model", "imagen-3.0-generate-002")

        if not self.api_key:
            raise ValueError("Google Image Generation requires an api_key in config")

        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        try:
            logger.info("Generating image with Google Imagen model: %s", self.model)
            result = self.client.models.generate_images(
                model=self.model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    output_mime_type="image/jpeg",
                    aspect_ratio="1:1",
                ),
            )

            if not result.generated_images:
                return "Failed to generate image: No image returned."

            generated_image = result.generated_images[0]
            # Since LoreKeeper is likely running locally, we can return a
            # base64 markdown embed or save it and return a local path.
            # For simplicity, we return base64 embedded markdown.
            b64_data = base64.b64encode(generated_image.image.image_bytes).decode(
                "utf-8"
            )
            return f"![Generated Image](data:image/jpeg;base64,{b64_data})"

        except Exception as e:
            logger.error("Error generating image with Google: %s", e)
            return f"Error generating image: {e}"
