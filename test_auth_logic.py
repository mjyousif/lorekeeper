import sys
from unittest.mock import MagicMock

class MockUser:
    def __init__(self, id):
        self.id = id

class MockChat:
    def __init__(self, id):
        self.id = id

class MockUpdate:
    def __init__(self, user_id=None, chat_id=None):
        self.effective_user = MockUser(user_id) if user_id else None
        self.effective_chat = MockChat(chat_id) if chat_id else None

import os
os.environ['ALLOWED_CHAT_IDS'] = '-12345'
os.environ['ALLOWED_USER_IDS'] = ''

sys.modules['telegram'] = MagicMock()
sys.modules['telegram.ext'] = MagicMock()
sys.modules['src.wrapper'] = MagicMock()
sys.modules['src.session_storage'] = MagicMock()
sys.modules['telegramify_markdown'] = MagicMock()

import src.telegram_bot as tb

print("Allowed User IDs:", tb.ALLOWED_USER_IDS)
print("Allowed Chat IDs:", tb.ALLOWED_CHAT_IDS)

print("Test (chat in list, user not in list):", tb.is_authorized(MockUpdate(111, -12345)))
