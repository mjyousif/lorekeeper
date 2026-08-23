import pytest

from src.core.tools.music_generation import get_music_generation_tool


def test_get_music_generation_tool_google():
    """Test getting the Google music generation tool."""
    config = {"provider": "google", "api_key": "test_key"}
    schema, impl = get_music_generation_tool(config)

    assert schema["function"]["name"] == "generate_music"
    assert "prompt" in schema["function"]["parameters"]["properties"]

    # Since it's just a stub returning a string
    result = impl("a relaxing melody")
    assert "Google music generation requested for prompt" in result


def test_get_music_generation_tool_google_missing_key():
    """Test missing API key for Google provider."""
    config = {"provider": "google"}
    with pytest.raises(ValueError, match="requires an api_key"):
        get_music_generation_tool(config)


def test_get_music_generation_tool_comfyui():
    """Test getting the ComfyUI music generation tool."""
    config = {"provider": "comfyui", "url": "http://localhost:8188"}
    schema, impl = get_music_generation_tool(config)

    assert schema["function"]["name"] == "generate_music"

    result = impl("an epic orchestral theme")
    assert "ComfyUI music generation requested for prompt" in result


def test_get_music_generation_tool_unknown():
    """Test unknown provider."""
    config = {"provider": "unknown_provider"}
    with pytest.raises(
        ValueError, match="Unknown music generation provider: unknown_provider"
    ):
        get_music_generation_tool(config)
