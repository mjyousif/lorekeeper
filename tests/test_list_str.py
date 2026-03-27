

def test_list_str(monkeypatch):
    from src.config import Config

    c = Config(allowed_chat_ids=["-1001234", "-567890"])
    assert -1001234 in c.allowed_chat_ids
    assert -567890 in c.allowed_chat_ids
    assert all(isinstance(x, int) for x in c.allowed_chat_ids)
