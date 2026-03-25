import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import sys
import importlib.util
import os

# Mock all dependencies
sys.modules['telegram'] = MagicMock()
sys.modules['telegram.ext'] = MagicMock()
sys.modules['telegramify_markdown'] = MagicMock()
sys.modules['pypdf'] = MagicMock()
sys.modules['chromadb'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['litellm'] = MagicMock()
sys.modules['pydantic-settings'] = MagicMock()
sys.modules['src.config'] = MagicMock()
sys.modules['src.wrapper'] = MagicMock()
sys.modules['src.vector_store'] = MagicMock()

# Load src/telegram_bot.py directly
module_name = 'src.telegram_bot'
file_path = os.path.abspath('src/telegram_bot.py')
spec = importlib.util.spec_from_file_location(module_name, file_path)
telegram_bot = importlib.util.module_from_spec(spec)
sys.modules[module_name] = telegram_bot

with patch('src.config.get_config'), \
     patch('src.wrapper.RAGWrapper'), \
     patch('sqlite3.connect'):
    spec.loader.exec_module(telegram_bot)

class TestTelegramBotAsync(unittest.IsolatedAsyncioTestCase):
    async def test_handle_message_success(self):
        # Setup mocks
        update = MagicMock()
        update.effective_chat.id = 123
        update.message.text = "Hello"
        update.message.chat.send_action = AsyncMock()
        update.message.reply_text = AsyncMock()

        context = MagicMock()

        # Mocking get_wrapper and its chat method
        mock_wrapper = MagicMock()
        mock_wrapper.sessions = {}
        mock_wrapper.chat.return_value = {"message": "Mock response"}

        with patch.object(telegram_bot, 'is_authorized', return_value=True), \
             patch.object(telegram_bot, 'get_wrapper', return_value=mock_wrapper), \
             patch.object(telegram_bot, 'get_history', return_value=[]) as mock_get_history, \
             patch.object(telegram_bot, 'set_history') as mock_set_history, \
             patch.object(telegram_bot, 'convert', return_value=("Mock response", [])):

            # Call the handler
            await telegram_bot.handle_message(update, context)

            # Verify get_history was called (wrapped in to_thread)
            mock_get_history.assert_called_once_with(123)

            # Verify set_history was called (wrapped in to_thread)
            mock_set_history.assert_called_once()

            # Verify reply was sent
            update.message.reply_text.assert_called_once()
            args, _ = update.message.reply_text.call_args
            self.assertEqual(args[0], "Mock response")

    async def test_handle_message_unauthorized(self):
        update = MagicMock()
        update.effective_user.id = 456
        update.effective_chat.id = 123
        update.message.reply_text = AsyncMock()
        context = MagicMock()

        with patch.object(telegram_bot, 'is_authorized', return_value=False):
            await telegram_bot.handle_message(update, context)
            update.message.reply_text.assert_called_with("❌ You are not authorized to use this bot.")

if __name__ == '__main__':
    unittest.main()
