import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


class SessionStorage:
    """Handles storing and retrieving conversation history using SQLite."""

    def __init__(self, db_path: str):
        """Initialize the storage and create the table if it does not exist.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the history table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        chat_id INTEGER PRIMARY KEY,
                        messages TEXT
                    )
                """)
            logger.info("Initialized SQLite session DB at %s", self.db_path)
        except Exception as e:
            logger.error("Failed to initialize DB: %s", e)

    def get_history(self, chat_id: int) -> list[dict]:
        """Retrieve the conversation history for a given chat ID.

        Args:
            chat_id: The unique identifier for the chat.

        Returns:
            A list of message dictionaries (role and content).
        """
        try:
            with sqlite3.connect(self.db_path) as con:
                cur = con.execute(
                    "SELECT messages FROM history WHERE chat_id=?", (chat_id,)
                )
                row = cur.fetchone()
                return json.loads(row[0]) if row else []
        except Exception as e:
            logger.error("Error reading history for chat_id %d: %s", chat_id, e)
            return []

    def set_history(self, chat_id: int, messages: list[dict]) -> None:
        """Store the conversation history for a given chat ID.

        Args:
            chat_id: The unique identifier for the chat.
            messages: A list of message dictionaries to store.
        """
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute(
                    "INSERT OR REPLACE INTO history (chat_id, messages) VALUES (?, ?)",
                    (chat_id, json.dumps(messages)),
                )
        except Exception as e:
            logger.error("Error saving history for chat_id %d: %s", chat_id, e)
