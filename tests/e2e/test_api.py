"""End-to-End Component Tests for the FastAPI Interface."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.config import Config
from src.interfaces.api import app, get_config


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
        files=str(data_dir),
        llm={"model": "test-model", "api_key": "test-key"},
        tools_config={},
    )


@pytest.fixture
def client(e2e_config):
    # Override get_config to return our e2e_config
    app.dependency_overrides[get_config] = lambda: e2e_config

    # We also need to clear the lru_cache on get_lorekeeper
    # to ensure it picks up the new config
    from src.interfaces.api import get_lorekeeper

    get_lorekeeper.cache_clear()

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@patch("src.core.chat_manager.litellm.completion")
def test_api_e2e_chat_simple(mock_completion, client):
    """Test a simple API chat interaction."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Greetings, traveler!"
    mock_choice.message.tool_calls = None
    mock_completion.return_value.choices = [mock_choice]

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Greetings, traveler!"
    assert mock_completion.called


@patch("src.core.chat_manager.litellm.completion")
def test_api_e2e_chat_memory_search(mock_completion, client):
    """Test an API interaction where the LLM uses the memory_search tool."""
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

    request_data = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Tell me about the sword of Eldoria."}
        ],
    }

    response = client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "It glows blue near orcs."
    assert mock_completion.call_count == 2
