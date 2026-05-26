import os
import pytest
import sqlite3
from src.auth_storage import AuthStorage

@pytest.fixture
def auth_storage(tmp_path):
    db_path = tmp_path / "test_auth.db"
    storage = AuthStorage(db_path=str(db_path))
    yield storage
    if db_path.exists():
        db_path.unlink()

def test_init_db(auth_storage):
    assert os.path.exists(auth_storage.db_path)
    with sqlite3.connect(auth_storage.db_path) as con:
        cur = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        assert "pending_pairs" in tables
        assert "authorized_users" in tables
        assert "authorized_chats" in tables

def test_user_pairing_flow(auth_storage):
    user_id = 12345

    # User shouldn't be authorized initially
    assert not auth_storage.is_user_authorized(user_id)

    # Create pending pair
    code = auth_storage.create_pending_pair(user_id)
    assert len(code) == 8

    # Approve pair
    result = auth_storage.approve_pair(code)
    assert result is not None
    assert result["type"] == "user"
    assert result["user_id"] == user_id

    # User should now be authorized
    assert auth_storage.is_user_authorized(user_id)

    # The pending pair should be removed
    result_again = auth_storage.approve_pair(code)
    assert result_again is None

def test_chat_pairing_flow(auth_storage):
    user_id = 12345
    chat_id = -98765

    # Chat shouldn't be authorized initially
    assert not auth_storage.is_chat_authorized(chat_id)

    # Create pending pair for a chat
    code = auth_storage.create_pending_pair(user_id, chat_id)
    assert len(code) == 8

    # Approve pair
    result = auth_storage.approve_pair(code)
    assert result is not None
    assert result["type"] == "chat"
    assert result["user_id"] == user_id
    assert result["chat_id"] == chat_id

    # Chat should now be authorized
    assert auth_storage.is_chat_authorized(chat_id)

    # Pending pair should be removed
    assert auth_storage.approve_pair(code) is None

def test_approve_invalid_code(auth_storage):
    assert auth_storage.approve_pair("INVALID") is None
