import logging
import sys

from src.core.config import get_config
from src.storage.auth_storage import AuthStorage

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 2:
        logger.error("Usage: uv run python -m scripts.approve_pair <CODE>")
        sys.exit(1)

    code = sys.argv[1]

    config = get_config()
    telegram_cfg = config.telegram or {}
    db_path = telegram_cfg.get("session_db", "sessions.db")

    auth_storage = AuthStorage(db_path=db_path)

    try:
        result = auth_storage.approve_pair(code)
        if result:
            if result["type"] == "user":
                logger.info("✅ Successfully authorized user %s", result["user_id"])
            elif result["type"] == "chat":
                logger.info(
                    "✅ Successfully authorized chat %s (requested by user %s)",
                    result["chat_id"],
                    result["user_id"],
                )
        else:
            logger.error("❌ Invalid or expired pairing code: %s", code)
            sys.exit(1)
    except Exception as e:
        logger.error("❌ An error occurred while approving the code: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
