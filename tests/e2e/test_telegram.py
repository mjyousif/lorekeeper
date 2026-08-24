"""End-to-End Component Tests for the Telegram Bot Interface."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Chat, Message, Update, User

from src.core.config import Config
from src.core.wrapper import LoreKeeper
from src.interfaces import telegram_bot


@pytest.fixture
def e2e_config(tmp_path):
    """Fixture to create a Config with test overrides for E2E tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "lore.txt").write_text(
        "The hero defeated the dragon using a magic shield."
    )

    return Config(
        db_path=str(tmp_path / "db"),
        files=str(data_dir),
        llm={"model": "test-model", "api_key": "test-key"},
        tools_config={},
        telegram={"bot_token": "fake_token", "allowed_user_ids": [123]},
    )


@pytest.fixture
def override_telegram_wrapper(e2e_config, monkeypatch):
    """Override the LoreKeeper singleton in the telegram module."""
    wrapper = LoreKeeper(config=e2e_config)
    monkeypatch.setattr(telegram_bot, "get_wrapper", lambda: wrapper)

    # Also patch authorization explicitly for test user 123
    monkeypatch.setattr(telegram_bot, "ALLOWED_USER_IDS", {123})
    return wrapper


@pytest.fixture
def mock_update():
    """Create a mock Telegram Update."""
    user = User(id=123, first_name="Test", is_bot=False, username="testuser")
    update = MagicMock(spec=Update)
    update.effective_user = user
    update.effective_chat = MagicMock(spec=Chat)
    update.effective_chat.id = 456
    update.effective_chat.type = "private"
    update.message = MagicMock(spec=Message)
    update.message.text = "Hello!"
    update.message.chat = update.effective_chat
    update.message.from_user = user
    update.message.reply_text = AsyncMock()
    update.message.chat.send_action = AsyncMock()
    return update


@pytest.fixture
def mock_context():
    """Create a mock Telegram context."""
    context = MagicMock()
    context.bot = MagicMock()
    context.bot.id = 999
    context.bot.username = "testbot"
    return context


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@patch("src.core.chat_manager.litellm.completion")
async def test_telegram_e2e_chat_simple(
    mock_completion, override_telegram_wrapper, mock_update, mock_context
):
    """Test a simple Telegram chat message interaction."""
    # Setup LLM mock response
    mock_choice = MagicMock()
    mock_choice.message.content = "Greetings from Telegram!"
    mock_choice.message.tool_calls = None
    mock_completion.return_value.choices = [mock_choice]

    mock_update.message.text = "Hello there"

    await telegram_bot.handle_message(mock_update, mock_context)

    # Verify LLM was called
    assert mock_completion.called

    # Verify the reply was sent back to Telegram
    mock_update.message.reply_text.assert_called_once()
    args, kwargs = mock_update.message.reply_text.call_args
    assert "Greetings from Telegram!" in args[0]


@pytest.mark.anyio
@patch("src.core.chat_manager.litellm.completion")
async def test_telegram_e2e_chat_memory_search(
    mock_completion, override_telegram_wrapper, mock_update, mock_context
):
    """Test a Telegram interaction where the LLM uses the memory_search tool."""
    # Setup LLM first response: invoke memory_search tool
    tool_call = MagicMock()
    tool_call.id = "call_lore"
    tool_call.function.name = "memory_search"
    tool_call.function.arguments = json.dumps({"query": "dragon shield"})

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
    final_choice.message.content = "A magic shield was used."
    final_choice.message.tool_calls = None

    mock_completion.side_effect = [
        MagicMock(choices=[tool_choice]),
        MagicMock(choices=[final_choice]),
    ]

    mock_update.message.text = "How was the dragon defeated?"

    await telegram_bot.handle_message(mock_update, mock_context)

    # Verify LLM was called twice (once for tool call, once for final answer)
    assert mock_completion.call_count == 2

    # Verify the reply was sent back to Telegram
    mock_update.message.reply_text.assert_called_once()
    args, kwargs = mock_update.message.reply_text.call_args
    assert "A magic shield was used." in args[0]
