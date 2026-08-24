from unittest.mock import patch

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


class MockResponse:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.data


@patch("src.core.tools.music_generation.comfyui.urllib.request.urlopen")
def test_get_music_generation_tool_comfyui(mock_urlopen):
    """Test getting the ComfyUI music generation tool."""
    mock_urlopen.side_effect = [
        MockResponse(b'{"prompt_id": "12345"}'),
        MockResponse(
            b'{"12345": {"outputs": {"107": {"audio": '
            b'[{"filename": "song.mp3", "subfolder": "", "type": "output"}]}}}}'
        ),
    ]
    config = {
        "provider": "comfyui",
        "url": "http://localhost:8188",
        "workflow_file": "dummy.json",
    }

    with patch(
        "builtins.open",
        return_value=MockResponse(
            '{"1": {"class_type": "TextEncode", "inputs": {"text": "dummy"}}}'
        ),
    ):
        schema, impl = get_music_generation_tool(config)

    assert schema["function"]["name"] == "generate_music"

    with patch(
        "builtins.open",
        return_value=MockResponse(
            '{"1": {"class_type": "TextEncode", "inputs": {"text": "dummy"}}}'
        ),
    ):
        result = impl("an epic orchestral theme")
    assert "[Generated Audio]" in result
    assert "song.mp3" in result


def test_get_music_generation_tool_unknown():
    """Test unknown provider."""
    config = {"provider": "unknown_provider"}
    with pytest.raises(
        ValueError, match="Unknown music generation provider: unknown_provider"
    ):
        get_music_generation_tool(config)
