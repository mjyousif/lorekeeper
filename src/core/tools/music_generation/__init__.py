from typing import Any, Callable, Dict, Tuple

from .base import MusicGenerationProvider
from .comfyui import ComfyUIMusicGenerationProvider
from .google import GoogleMusicGenerationProvider


def get_music_generation_tool(
    config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Callable]:
    """Factory to create the music generation tool schema and implementation."""

    provider_name = config.get("provider", "google").lower()

    provider: MusicGenerationProvider

    if provider_name == "google":
        provider = GoogleMusicGenerationProvider(config)
    elif provider_name == "comfyui":
        provider = ComfyUIMusicGenerationProvider(config)
    else:
        raise ValueError(f"Unknown music generation provider: {provider_name}")

    schema = {
        "type": "function",
        "function": {
            "name": "generate_music",
            "description": (
                "Generate music or audio based on a text prompt. "
                "Use this tool when the user asks you to create, "
                "compose, or generate music or audio. "
                "By default, you MUST generate and provide original "
                "lyrics for the song, unless the user explicitly "
                "requests an instrumental. IMPORTANT: You MUST print "
                "the generated lyrics in your final response to the user!"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "A detailed description of the music or "
                            "audio to generate (genre, mood, instruments)."
                        ),
                    },
                    "lyrics": {
                        "type": "string",
                        "description": (
                            "The lyrics for the song. You must ALWAYS "
                            "generate and provide full, original lyrics "
                            "here unless the user explicitly asked "
                            "for an instrumental."
                        ),
                    },
                },
                "required": ["prompt", "lyrics"],
            },
        },
    }

    def impl(prompt: str, lyrics: str = "") -> str:
        return provider.generate(prompt, lyrics)

    return schema, impl
