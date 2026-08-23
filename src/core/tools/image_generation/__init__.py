from typing import Any, Callable, Dict, Tuple

from .base import ImageGenerationProvider
from .comfyui import ComfyUIImageGenerationProvider
from .google import GoogleImageGenerationProvider


def get_image_generation_tool(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Callable]:
    """Factory to create the image generation tool schema and implementation."""

    provider_name = config.get("provider", "google").lower()

    provider: ImageGenerationProvider

    if provider_name == "google":
        provider = GoogleImageGenerationProvider(config)
    elif provider_name == "comfyui":
        provider = ComfyUIImageGenerationProvider(config)
    else:
        raise ValueError(f"Unknown image generation provider: {provider_name}")

    schema = {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image based on a text prompt. "
                "Use this tool when the user asks you to create, "
                "draw, or generate an image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A detailed description of the image to generate."
                        ),
                    }
                },
                "required": ["prompt"],
            },
        },
    }

    def impl(prompt: str) -> str:
        return provider.generate(prompt)

    return schema, impl
