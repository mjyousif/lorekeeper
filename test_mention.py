import asyncio
from telegram import Message, User, Chat, Update, MessageEntity
from telegram.ext import filters
from datetime import datetime

async def main():
    u = User(id=1, first_name="Test", is_bot=False)
    c = Chat(id=1, type="group")

    # Mention is treated as COMMAND?
    m_command = Message(
        message_id=2,
        date=datetime.now(),
        chat=c,
        text="/start",
        entities=[MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=6)]
    )

    m_mention = Message(
        message_id=1,
        date=datetime.now(),
        chat=c,
        text="@mybot hello",
        entities=[MessageEntity(type=MessageEntity.MENTION, offset=0, length=6)]
    )

    f_cmd = filters.COMMAND
    f_text = filters.TEXT

    up_cmd = Update(update_id=1, message=m_command)
    up_ment = Update(update_id=2, message=m_mention)

    print("Is '/start' a command?", f_cmd.check_update(up_cmd))
    print("Is '@mybot hello' a command?", f_cmd.check_update(up_ment))

    f_combined = filters.TEXT & ~filters.COMMAND
    print("Is '@mybot hello' passing combined filter?", f_combined.check_update(up_ment))

asyncio.run(main())
