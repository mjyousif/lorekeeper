import asyncio
from datetime import datetime

from telegram import Chat, Message, MessageEntity, Update, User
from telegram.ext import filters


async def main():
    User(id=1, first_name="Test", is_bot=False)
    c = Chat(id=-1001234, type="group")

    # Text mention
    m_mention = Message(
        message_id=1,
        date=datetime.now(),
        chat=c,
        text="@mybot hello",
        entities=[MessageEntity(type=MessageEntity.MENTION, offset=0, length=6)],
    )

    up = Update(update_id=1, message=m_mention)

    # Are we accidentally filtering out groups in filters.TEXT?
    f = filters.UpdateType.MESSAGE
    print("Does it pass UpdateType.MESSAGE?", f.check_update(up))


asyncio.run(main())
