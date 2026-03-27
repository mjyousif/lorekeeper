from unittest.mock import MagicMock, patch


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


def test_auth_logic(monkeypatch):
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "-12345")
    monkeypatch.setenv("ALLOWED_USER_IDS", "")

    # Needs to match what get_config().telegram.get("bot_token") returns
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy-token")

    with patch.dict(
        "sys.modules",
        {
            "telegram": MagicMock(),
            "telegram.ext": MagicMock(),
            "src.wrapper": MagicMock(),
            "src.session_storage": MagicMock(),
            "telegramify_markdown": MagicMock(),
        },
    ):
        # Reset get_config cache to load the newly set env vars
        from src.config import get_config

        get_config.cache_clear()

        import src.telegram_bot as tb

        assert tb.ALLOWED_USER_IDS == set()
        assert tb.ALLOWED_CHAT_IDS == {-12345}

        assert tb.is_authorized(MockUpdate(111, -12345)) is True
