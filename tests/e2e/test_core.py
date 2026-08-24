"""End-to-End Component Tests for LoreKeeper."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Config
from src.core.wrapper import LoreKeeper


@pytest.fixture
def e2e_config(tmp_path):
    """Fixture to create a Config with test overrides for E2E tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "lore.txt").write_text(
        "The ancient sword of Eldoria glows blue in the presence of orcs."
    )

    return Config(
        db_path=str(tmp_path / "db"),
        llm={"model": "test-model", "api_key": "test-key"},
        tools_config={
            "image_generation": {"provider": "google", "api_key": "fake_img_key"},
            "music_generation": {"provider": "google", "api_key": "fake_music_key"},
        },
    )


@patch("src.core.chat_manager.litellm.completion")
@patch("src.core.tools.image_generation.google.genai")
def test_e2e_happy_path_chat_simple(mock_genai, mock_completion, e2e_config, tmp_path):
    """Test a simple chat interaction where the LLM responds directly."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Greetings, traveler!"
    mock_choice.message.tool_calls = None
    mock_completion.return_value.choices = [mock_choice]

    wrapper = LoreKeeper(config=e2e_config, files=str(tmp_path / "data"))

    response = wrapper.chat(session_id="session1", message="Hello!")

    assert response["message"] == "Greetings, traveler!"
    assert mock_completion.called
    messages = mock_completion.call_args[1]["messages"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Hello!"


@patch("src.core.chat_manager.litellm.completion")
@patch("src.core.tools.image_generation.google.genai")
def test_e2e_happy_path_memory_search(
    mock_genai, mock_completion, e2e_config, tmp_path
):
    """Test an interaction where the LLM uses the memory_search tool."""
    wrapper = LoreKeeper(config=e2e_config, files=str(tmp_path / "data"))

    # Setup LLM first response: invoke memory_search tool
    tool_call = MagicMock()
    tool_call.id = "call_lore"
    tool_call.function.name = "memory_search"
    tool_call.function.arguments = json.dumps({"query": "ancient sword of Eldoria"})

    tool_choice = MagicMock()
    tool_choice.message.content = None
    tool_choice.message.tool_calls = [tool_call]
    tool_choice.message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_lore",
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }

    # Setup LLM second response: answer using context
    final_choice = MagicMock()
    final_choice.message.content = "It glows blue near orcs."
    final_choice.message.tool_calls = None

    mock_completion.side_effect = [
        MagicMock(choices=[tool_choice]),
        MagicMock(choices=[final_choice]),
    ]

    response = wrapper.chat(
        session_id="session_lore", message="Tell me about the sword of Eldoria."
    )

    assert response["message"] == "It glows blue near orcs."
    assert mock_completion.call_count == 2

    # Verify tool response was injected
    messages_second_call = mock_completion.call_args_list[1][1]["messages"]
    tool_msg = messages_second_call[-1]
    assert tool_msg["role"] == "tool"
    assert tool_msg["name"] == "memory_search"
    assert "glows blue in the presence of orcs" in tool_msg["content"]


@patch("src.core.chat_manager.litellm.completion")
@patch("src.core.tools.image_generation.google.genai")
def test_e2e_happy_path_image_generation(
    mock_genai, mock_completion, e2e_config, tmp_path
):
    """Test an interaction where the LLM uses the generate_image tool."""
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_image = MagicMock()
    mock_image.image.image_bytes = b"fake_image_bytes"
    mock_result = MagicMock()
    mock_result.generated_images = [mock_image]
    mock_client.models.generate_images.return_value = mock_result

    wrapper = LoreKeeper(config=e2e_config, files=str(tmp_path / "data"))

    # Setup LLM first response: invoke generate_image tool
    tool_call = MagicMock()
    tool_call.id = "call_img"
    tool_call.function.name = "generate_image"
    tool_call.function.arguments = json.dumps({"prompt": "A blue glowing sword"})

    tool_choice = MagicMock()
    tool_choice.message.content = None
    tool_choice.message.tool_calls = [tool_call]
    tool_choice.message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_img",
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }

    # Setup LLM second response: return the image markdown
    final_choice = MagicMock()
    final_choice.message.content = (
        "Here is the image you requested! "
        "![Generated Image](data:image/jpeg;base64,ZmFrZV9pbWFnZV9ieXRlcw==)"
    )
    final_choice.message.tool_calls = None

    mock_completion.side_effect = [
        MagicMock(choices=[tool_choice]),
        MagicMock(choices=[final_choice]),
    ]

    response = wrapper.chat(session_id="session_img", message="Show me the sword.")

    assert "![Generated Image]" in response["message"]
    assert "ZmFrZV9pbWFnZV9ieXRlcw==" in response["message"]
    assert mock_completion.call_count == 2
    mock_client.models.generate_images.assert_called_once()


@patch("src.core.chat_manager.litellm.completion")
@patch("src.core.tools.image_generation.google.genai")
def test_e2e_happy_path_music_generation(
    mock_genai, mock_completion, e2e_config, tmp_path
):
    """Test an interaction where the LLM uses the generate_music tool."""
    wrapper = LoreKeeper(config=e2e_config, files=str(tmp_path / "data"))

    # Setup LLM first response: invoke generate_music tool
    tool_call = MagicMock()
    tool_call.id = "call_music"
    tool_call.function.name = "generate_music"
    tool_call.function.arguments = json.dumps({"prompt": "epic battle music"})

    tool_choice = MagicMock()
    tool_choice.message.content = None
    tool_choice.message.tool_calls = [tool_call]
    tool_choice.message.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_music",
                "type": "function",
                "function": {
                    "name": "generate_music",
                    "arguments": tool_call.function.arguments,
                },
            }
        ],
    }

    # Setup LLM second response: return the music text
    final_choice = MagicMock()
    final_choice.message.content = (
        "Here is your music: Google music generation requested for prompt: "
        "epic battle music"
    )
    final_choice.message.tool_calls = None

    mock_completion.side_effect = [
        MagicMock(choices=[tool_choice]),
        MagicMock(choices=[final_choice]),
    ]

    response = wrapper.chat(
        session_id="session_music", message="Play some battle music."
    )

    assert "Google music generation requested" in response["message"]
    assert mock_completion.call_count == 2
