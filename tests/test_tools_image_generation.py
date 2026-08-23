from unittest.mock import MagicMock, patch

import pytest

from src.core.tools.image_generation import get_image_generation_tool
from src.core.tools.image_generation.comfyui import ComfyUIImageGenerationProvider
from src.core.tools.image_generation.google import GoogleImageGenerationProvider


def test_get_image_generation_tool_google():
    config = {"provider": "google", "api_key": "fake_key"}
    schema, impl = get_image_generation_tool(config)

    assert schema["function"]["name"] == "generate_image"
    assert "prompt" in schema["function"]["parameters"]["required"]

    # We shouldn't call the real implementation without mocking the client,
    # but we can verify the provider is the correct type.
    # The impl is a closure over the provider, so we can't easily check the
    # provider type without inspecting the closure or just testing the providers
    # directly below.


def test_get_image_generation_tool_comfyui():
    config = {"provider": "comfyui", "url": "http://test:8188"}
    schema, impl = get_image_generation_tool(config)
    assert schema["function"]["name"] == "generate_image"

    result = impl("test prompt")
    assert "ComfyUI image generation requested" in result
    assert "test prompt" in result


def test_get_image_generation_tool_unknown():
    config = {"provider": "unknown_provider"}
    with pytest.raises(
        ValueError, match="Unknown image generation provider: unknown_provider"
    ):
        get_image_generation_tool(config)


def test_comfyui_provider():
    provider = ComfyUIImageGenerationProvider({"url": "http://test:8188"})
    assert provider.url == "http://test:8188"
    result = provider.generate("test prompt")
    assert "test prompt" in result
    assert "ComfyUI" in result


@patch("src.core.tools.image_generation.google.genai")
def test_google_provider_init(mock_genai):
    # Test valid init
    provider = GoogleImageGenerationProvider(
        {"api_key": "test_key", "model": "test-model"}
    )
    assert provider.api_key == "test_key"
    assert provider.model == "test-model"
    mock_genai.Client.assert_called_once_with(api_key="test_key")


def test_google_provider_missing_key():
    with pytest.raises(ValueError, match="Google Image Generation requires an api_key"):
        GoogleImageGenerationProvider({})


@patch("src.core.tools.image_generation.google.genai")
def test_google_provider_generate(mock_genai):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    # Create a mock response
    mock_image = MagicMock()
    mock_image.image.image_bytes = b"fake_image_bytes"
    mock_result = MagicMock()
    mock_result.generated_images = [mock_image]

    mock_client.models.generate_images.return_value = mock_result

    provider = GoogleImageGenerationProvider({"api_key": "test_key"})
    result = provider.generate("test prompt")

    assert result.startswith("![Generated Image](data:image/jpeg;base64,")
    # Base64 encoding of "fake_image_bytes"
    import base64

    expected_b64 = base64.b64encode(b"fake_image_bytes").decode("utf-8")
    assert expected_b64 in result

    mock_client.models.generate_images.assert_called_once()
    args, kwargs = mock_client.models.generate_images.call_args
    assert kwargs["model"] == "imagen-3.0-generate-002"
    assert kwargs["prompt"] == "test prompt"


@patch("src.core.tools.image_generation.google.genai")
def test_google_provider_generate_no_images(mock_genai):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_result = MagicMock()
    mock_result.generated_images = []
    mock_client.models.generate_images.return_value = mock_result

    provider = GoogleImageGenerationProvider({"api_key": "test_key"})
    result = provider.generate("test prompt")

    assert result == "Failed to generate image: No image returned."


@patch("src.core.tools.image_generation.google.genai")
def test_google_provider_generate_error(mock_genai):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_client.models.generate_images.side_effect = Exception("API error")

    provider = GoogleImageGenerationProvider({"api_key": "test_key"})
    result = provider.generate("test prompt")

    assert "Error generating image: API error" in result
