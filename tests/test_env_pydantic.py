

def test_env_pydantic(monkeypatch):
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "-100123")
    monkeypatch.setenv("ALLOWED_USER_IDS", "456")

    from src.config import Config

    c = Config()

    assert -100123 in c.allowed_chat_ids
    assert isinstance(list(c.allowed_chat_ids)[0], int)

    monkeypatch.setenv("ALLOWED_CHAT_IDS", "-100123,-456789")
    c2 = Config()
    assert -100123 in c2.allowed_chat_ids
    assert -456789 in c2.allowed_chat_ids
    assert isinstance(list(c2.allowed_chat_ids)[0], int)
