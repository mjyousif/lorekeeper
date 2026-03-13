# Getting Telegram User and Chat IDs

To authorize specific users or chats, you need their numeric IDs.

## Get Your User ID

1. Open **https://web.telegram.org** in your browser
2. Open **Saved Messages** (or any private chat)
3. Look at the browser's address bar. It will look like:
   ```
   https://web.telegram.org/a/#123456789
   ```
   The number after `#` is your **user ID**.

   If you don't see the `#` segment, add it manually: after opening the chat, the URL updates to include the peer ID.

## Get a Chat/Group ID

1. Open **https://web.telegram.org** in your browser
2. Open the target group/chat in the left sidebar
3. Look at the browser's address bar. It will look like:
   ```
   https://web.telegram.org/a/#-1001234567890
   ```
   The number after `#` is the **chat ID**. Group IDs are typically negative (start with `-100`).

  Tip: For groups, the URL may not show the `#` initially. Click the group name to open its info, then check the URL; it should update.

## Using IDs

Set environment variables (comma-separated):

```bash
ALLOWED_USER_IDS=123456789,987654321
ALLOWED_CHAT_IDS=-1001111111111,-1002222222222
```

Or in a config file (`config.yaml`):

```yaml
allowed_user_ids:
  - 123456789
  - 987654321
allowed_chat_ids:
  - -1001111111111
  - -1002222222222
```

**Note:** If neither `ALLOWED_USER_IDS` nor `ALLOWED_CHAT_IDS` is set, the bot will deny all access (secure default). If you set them but only `ALLOWED_USER_IDS`, chats are still checked separately; you may need to whitelist the group ID if you want the bot to work in a group.