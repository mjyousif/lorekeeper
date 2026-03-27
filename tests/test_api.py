import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api import app, get_lorekeeper
from src.wrapper import LoreKeeper


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "LoreKeeper API is running. POST to /v1/chat/completions to interact."
    }


def test_chat_completions_success(client):
    mock_rag = MagicMock(spec=LoreKeeper)
    mock_rag.chat.return_value = {
        "message": "Mocked LLM response",
        "context": ["mocked context"],
    }

    app.dependency_overrides[get_lorekeeper] = lambda: mock_rag

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)

    # Assert dependency override worked and mock was called
    mock_rag.chat.assert_called_once_with(
        session_id="api_session_placeholder", message="Hello"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "test-model"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Mocked LLM response"
    assert data["context"] == ["mocked context"]
    assert "id" in data
    assert data["object"] == "chat.completion"


def test_chat_completions_no_context(client):
    mock_rag = MagicMock(spec=LoreKeeper)
    mock_rag.chat.return_value = {
        "message": "Mocked LLM response",
        "context": [],  # Context could be empty
    }

    app.dependency_overrides[get_lorekeeper] = lambda: mock_rag

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["context"] is None


def test_chat_completions_empty_messages(client):
    request_data = {"model": "test-model", "messages": []}

    response = client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == 400
    assert response.json()["detail"] == "Messages list cannot be empty."


def test_chat_completions_rag_exception(client):
    mock_rag = MagicMock(spec=LoreKeeper)
    mock_rag.chat.side_effect = Exception("Something went wrong")

    app.dependency_overrides[get_lorekeeper] = lambda: mock_rag

    request_data = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    response = client.post("/v1/chat/completions", json=request_data)

    assert response.status_code == 500
    assert response.json()["detail"] == "RAG error: Something went wrong"


@patch("src.api.LoreKeeper")
@patch("src.api.get_config")
def test_get_lorekeeper_caching(mock_get_config, mock_lorekeeper_class):
    # Clear cache before testing to ensure clean state
    get_lorekeeper.cache_clear()

    mock_config = MagicMock()
    mock_get_config.return_value = mock_config

    mock_instance = MagicMock()
    mock_lorekeeper_class.return_value = mock_instance

    # Call multiple times
    instance1 = get_lorekeeper()
    instance2 = get_lorekeeper()
    instance3 = get_lorekeeper()

    # Assert get_config was called exactly once due to caching
    mock_get_config.assert_called_once()

    # Assert LoreKeeper was instantiated exactly once
    mock_lorekeeper_class.assert_called_once_with(mock_config)

    # Assert all returned instances are the same object
    assert instance1 is instance2 is instance3 is mock_instance
