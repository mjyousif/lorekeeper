import contextlib
import logging
import sqlite3
import contextlib
@contextlib.contextmanager
def _get_db(db_path):
    con = sqlite3.connect(db_path)
    try:
        with con:
            yield con
    finally:
        con.close()
import time
import uuid
from typing import Optional, Any

logger = logging.getLogger(__name__)


class AuthStorage:
    """Handles storing and retrieving dynamic authorization data using SQLite."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        try:
            with _get_db(self.db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS pending_pairs (
                        code TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        chat_id INTEGER,
                        created_at INTEGER NOT NULL
                    )
                """)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS authorized_users (
                        user_id INTEGER PRIMARY KEY,
                        approved_at INTEGER NOT NULL
                    )
                """)
                con.execute("""
                    CREATE TABLE IF NOT EXISTS authorized_chats (
                        chat_id INTEGER PRIMARY KEY,
                        approved_by INTEGER NOT NULL,
                        approved_at INTEGER NOT NULL
                    )
                """)
            logger.info("Initialized SQLite auth DB at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize auth DB: %s", e)

    def create_pending_pair(self, user_id: int, chat_id: Optional[int] = None) -> str:
        """Creates a pending pair request and returns the code."""
        code = str(uuid.uuid4())[:8].upper()
        now = int(time.time())
        try:
            with _get_db(self.db_path) as con:
                con.execute(
                    "INSERT INTO pending_pairs (code, user_id, chat_id, created_at) VALUES (?, ?, ?, ?)",
                    (code, user_id, chat_id, now),
                )
            return code
        except Exception as e:
            logger.error("Error creating pending pair for user %s: %s", user_id, e)
            raise e

    def approve_pair(self, code: str) -> dict[str, Any] | None:
        """Approves a pending pair request by code.
        Returns a dict with 'type' ('user' or 'chat') and the relevant IDs,
        or None if not found.
        """
        code = code.upper().strip()
        try:
            with _get_db(self.db_path) as con:
                cur = con.execute(
                    "SELECT user_id, chat_id FROM pending_pairs WHERE code = ?", (code,)
                )
                row = cur.fetchone()
                if not row:
                    return None

                user_id, chat_id = row
                now = int(time.time())

                if chat_id:
                    # It's a channel/group pairing
                    con.execute(
                        "INSERT OR REPLACE INTO authorized_chats (chat_id, approved_by, approved_at) VALUES (?, ?, ?)",
                        (chat_id, user_id, now),
                    )
                    con.execute("DELETE FROM pending_pairs WHERE code = ?", (code,))
                    return {"type": "chat", "user_id": user_id, "chat_id": chat_id}
                else:
                    # It's a user pairing
                    con.execute(
                        "INSERT OR REPLACE INTO authorized_users (user_id, approved_at) VALUES (?, ?)",
                        (user_id, now),
                    )
                    con.execute("DELETE FROM pending_pairs WHERE code = ?", (code,))
                    return {"type": "user", "user_id": user_id}
        except Exception as e:
            logger.error("Error approving pair for code %s: %s", code, e)
            raise e

    def is_user_authorized(self, user_id: int) -> bool:
        """Checks if a user is dynamically authorized."""
        try:
            with _get_db(self.db_path) as con:
                cur = con.execute(
                    "SELECT 1 FROM authorized_users WHERE user_id = ?", (user_id,)
                )
                return cur.fetchone() is not None
        except Exception as e:
            logger.error("Error checking auth for user %d: %s", user_id, e)
            return False

    def is_chat_authorized(self, chat_id: int) -> bool:
        """Checks if a chat is dynamically authorized."""
        try:
            with _get_db(self.db_path) as con:
                cur = con.execute(
                    "SELECT 1 FROM authorized_chats WHERE chat_id = ?", (chat_id,)
                )
                return cur.fetchone() is not None
        except Exception as e:
            logger.error("Error checking auth for chat %d: %s", chat_id, e)
            return False
