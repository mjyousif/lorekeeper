"""End-to-End Component Tests for the Gradio UI."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Config
from src.core.wrapper import LoreKeeper
from src.interfaces import gradio_app


@pytest.fixture
def e2e_config(tmp_path):
    """Fixture to create a Config with test overrides for E2E tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "lore.txt").write_text("The wizard lives in the tall tower.")

    return Config(
        db_path=str(tmp_path / "db"),
        files=str(data_dir),
        llm={"model": "test-model", "api_key": "test-key"},
        tools_config={},
    )


@pytest.fixture
def override_gradio_wrapper(e2e_config, monkeypatch):
    """Override the LoreKeeper singleton in the gradio module."""
    wrapper = LoreKeeper(config=e2e_config)
    monkeypatch.setattr(gradio_app, "get_wrapper", lambda: wrapper)
    return wrapper


@patch("src.core.chat_manager.litellm.completion")
def test_gradio_e2e_rag_query(mock_completion, override_gradio_wrapper):
    """Test the Gradio UI's rag_query integration with LoreKeeper and LLM."""

    # Setup LLM first response: invoke memory_search tool
    tool_call = MagicMock()
    tool_call.id = "call_lore"
    tool_call.function.name = "memory_search"
    tool_call.function.arguments = json.dumps({"query": "wizard tower"})

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
    final_choice.message.content = "He lives in the tall tower."
    final_choice.message.tool_calls = None

    mock_completion.side_effect = [
        MagicMock(choices=[tool_choice]),
        MagicMock(choices=[final_choice]),
    ]

    full_output, context_str, session_id, history_info = gradio_app.rag_query(
        query="Where does the wizard live?",
        session_id="test_ui_session",
        n_results=3,
        include_context=True,
        include_history=True,
    )

    assert "He lives in the tall tower." in full_output
    assert "wizard lives in the tall tower" in context_str
    assert "test_ui_session" == session_id
    assert (
        "Session has 4 messages" in history_info
    )  # System + User + ToolCall + ToolResponse
    assert mock_completion.call_count == 2
