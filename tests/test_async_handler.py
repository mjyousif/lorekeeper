import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import sys
import importlib.util
import os

def load_telegram_bot():
    # Helper to load src/telegram_bot.py with necessary mocks
    mocks = {
        'telegram': MagicMock(),
        'telegram.ext': MagicMock(),
        'telegramify_markdown': MagicMock(),
        'pypdf': MagicMock(),
        'chromadb': MagicMock(),
        'sentence_transformers': MagicMock(),
        'litellm': MagicMock(),
        'pydantic-settings': MagicMock(),
        'src.config': MagicMock(),
        'src.wrapper': MagicMock(),
        'src.vector_store': MagicMock(),
    }

    with patch.dict('sys.modules', mocks):
        module_name = 'src.telegram_bot'
        file_path = os.path.abspath('src/telegram_bot.py')
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        telegram_bot = importlib.util.module_from_spec(spec)

        with patch('src.config.get_config'), \
             patch('src.wrapper.RAGWrapper'), \
             patch('sqlite3.connect'):
            spec.loader.exec_module(telegram_bot)
        return telegram_bot

class TestTelegramBotAsync(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.telegram_bot = load_telegram_bot()

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

        with patch.object(self.telegram_bot, 'is_authorized', return_value=True), \
             patch.object(self.telegram_bot, 'get_wrapper', return_value=mock_wrapper), \
             patch.object(self.telegram_bot, 'get_history', return_value=[]) as mock_get_history, \
             patch.object(self.telegram_bot, 'set_history') as mock_set_history, \
             patch.object(self.telegram_bot, 'convert', return_value=("Mock response", [])):

            # Call the handler
            await self.telegram_bot.handle_message(update, context)

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

        with patch.object(self.telegram_bot, 'is_authorized', return_value=False):
            await self.telegram_bot.handle_message(update, context)
            update.message.reply_text.assert_called_with("❌ You are not authorized to use this bot.")

if __name__ == '__main__':
    unittest.main()
