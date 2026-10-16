# chatgpt-telegram-bot

![Demo](https://user-images.githubusercontent.com/10773481/235257879-ce91d3d3-769a-4089-addd-2ac9eea26f35.gif)

A Telegram bot that forwards messages to OpenAI-compatible and Anthropic APIs and streams the
reply back, editing the message as tokens arrive. One bot can serve any number of models and
endpoints side by side, each reachable through its own message prefix.

A fork of [zzh1996/chatgpt-telegram-bot](https://github.com/zzh1996/chatgpt-telegram-bot).

## Features

- **Streaming output** — the reply is edited into place as it is generated, rate-limited so a
  long generation stays inside Telegram's limits.
- **Many models at once** — every model in `bot.toml` gets a prefix. Three backends are
  supported: the OpenAI Responses API (`openai`, the default), Chat Completions
  (`openai_legacy`), and Anthropic (`anthropic`).
- **Per-message overrides** — thinking effort, web search, markdown rendering and the system
  prompt can all be changed inline, without touching the config.
- **Reasoning** — a model's thinking is shown in a collapsible quote above the answer, and is
  carried into later turns where the provider allows it.
- **Threaded, forkable history** — reply to the bot to continue a conversation; reply to an
  older message to fork it. History lives in a local database and survives restarts.
- **Mid-thread model switching** — a prefix on a reply moves the conversation to another model
  from that point on.
- **Output manipulation** — `/manipulate` rewrites what the bot is recorded as having said, so
  later turns build on your version.
- **Conversation import** — `/import` replays a conversation exported from elsewhere into the
  chat, one message per turn, each of them replyable.
- **Long replies** — output over Telegram's per-message limit is split across several messages.
- **Multimodal input** — photos (single or albums) and UTF-8 text files.
- **Concurrency and queueing** — requests run in parallel, and a reply to a message that is
  still generating waits for it to finish.
- **Per-chat allowlist** — the bot answers only in chats you list, and stays silent elsewhere.

## Setup

1. Get an API key for whichever provider you plan to use.
1. Create a bot with [BotFather](https://t.me/BotFather).
1. Get `api_id` and `api_hash` from <https://my.telegram.org/apps>.
1. Copy `bot.example.toml` to `bot.toml` and edit it to taste.
1. Set these environment variables:
   - `TELEGRAM_API_ID`
   - `TELEGRAM_API_HASH`
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY_{name}` for each endpoint `{name}` in `bot.toml`
   - `ALL_PROXY` (optional) — HTTP or SOCKS proxy for the Telegram connection
1. Run it: `uv run chatgpt-telegram-bot -c bot.toml`
1. Send `/ping` to the bot. It replies with the `chat_id` of the chat and whether that chat is
   allowed — put the ids you want in `allowed_chats`.

To use the bot in a group you must **disable privacy mode** via BotFather
(`/setprivacy` → Disable). With privacy mode on, Telegram delivers only commands and replies
to the bot, so ordinary prefixed messages never arrive and the bot looks dead.

## Talking to the bot

Start a conversation with a message beginning with a model's prefix, and continue it by
replying to the bot. A prefix on a reply switches the model from there on; without one, the
conversation stays on the model it was already using.

Overrides go between the prefix and the message, separated by commas:

| Syntax | Effect |
| --- | --- |
| `gpt hello` | ask the model behind prefix `gpt` |
| `gpt,t hello` | enable thinking at the backend's default effort |
| `gpt,t=high hello` | set thinking effort (`none`…`max`, or a token budget for Anthropic) |
| `gpt,t= hello` | disable thinking |
| `gpt,s hello` | enable web search (`openai` and `anthropic` backends) |
| `gpt,r hello` | let Telegram render the markdown server-side |
| `gpt,r= hello` | render to message entities on the client instead |
| `gpt,[be terse] hello` | replace the system prompt for this conversation |
| `gpt,+[be terse] hello` | append to the default system prompt |
| `gpt,[] hello` | send no system prompt at all |

Several can be combined (`gpt,t=high,s hello`), and brackets may contain spaces and escaped
brackets (`\[`, `\]`).

Commands:

- `/help` — the model list, the override syntax and the current system prompt
- `/ping` — chat id, user id, and whether the chat is allowed
- `/manipulate <text>`, or `/manipulate <think>reasoning</think> <text>` — reply to a bot
  message with this to rewrite what it is recorded as having said
- `/import`, or `/import <prefix>` — attach a JSON export (or reply to one) to replay a
  conversation into the chat

Attach a photo or a UTF-8 text file to a message, with or without a caption; you can also
reply to someone else's photo or file to feed it to a model.

`/import` takes a conversation exported as JSON:

```json
{
  "title": "Reading list",
  "messages": [
    { "role": "user", "content": "what should I read?" },
    { "role": "assistant", "content": "start with ..." }
  ]
}
```

Roles must alternate, starting with the user. The bot confirms the title and then posts one
message per turn, showing the prompt above the response; reply to any of them to continue the
conversation from that point.

Sending is paced to stay inside Telegram's rate limits, so a short import appears at once
while a very long one trickles in over several minutes; the confirmation says so when it will
take more than a minute.

An import records no model of its own, so that first reply has to carry a prefix to choose one
— whichever you pick continues from there on. Passing `/import <prefix>` sets it up front
instead, and then a plain reply is enough.

## Configuration

`bot.toml`, top level:

| Key | Default | Meaning |
| --- | --- | --- |
| `admin_id` | required | Telegram user id of the admin; always allowed |
| `allowed_chats` | `[]` | chat ids the bot answers in, besides the admin |
| `endpoints` | required | list of `{ name, url, default_suffix }` |
| `default_endpoint` | required | endpoint used by models that name none |
| `models` | required | list of models, see below |
| `system_prompt` | a generic one | supports `{current_date}` and `{model}` |
| `timezone` | `"UTC"` | zone for `{current_date}` |
| `default_image_prompt` | `"Describe the image"` | used when a photo arrives with no caption |
| `default_image_reply_prompt` | `"Continue"` | the same, mid-conversation |
| `image_cache_size` | 50 MB | on-disk cache for photos, in bytes |

Each entry in `models`:

| Key | Default | Meaning |
| --- | --- | --- |
| `prefix` | required | message prefix; no spaces or commas |
| `name` | required | model name sent to the API |
| `endpoint` | `default_endpoint` | which endpoint to call |
| `api_type` | `"openai"` | `openai`, `openai_legacy` or `anthropic` |
| `suffix` | endpoint's | path appended to the endpoint URL |
| `thinking` | `null` | effort level, token budget, or `"adaptive"` (Anthropic only) |
| `search` | `false` | enable web search |
| `richtext` | `false` | let Telegram render the markdown rather than sending entities |
| `system_prompt` | `null` | replaces the default prompt for this model |
| `system_prompt_append` | `null` | appended to the default prompt |
| `no_system_prompt` | `false` | send no system prompt |

Persistent state — the conversation database, the photo cache and the Telethon session — goes
to `$XDG_STATE_HOME/chatgpt-telegram-bot` (`~/.local/state/…` by default), or wherever
`--data-dir` points.

## Running

```bash
uv run chatgpt-telegram-bot -c bot.toml
uv run chatgpt-telegram-bot -c bot.toml --debug   # DEBUG log level
uv run chatgpt-telegram-bot -c bot.toml --trace   # DEBUG plus per-event stream logging
```

This flake exposes the package as `packages.default` and a NixOS module as
`nixosModules.default`, which runs the bot under `DynamicUser` with state in
`/var/lib/chatgpt-telegram-bot` and takes `configFile` and `envFile` options. `nix develop`
gives a shell with `uv`, `ruff` and `basedpyright`; `nix flake check` runs the tests, the type
checker and the formatter.

## Logging

When running under systemd the bot sends structured entries to journald over the native
protocol (via `systemd-python`), with proper syslog priorities and code locations. This is
detected through the `JOURNAL_STREAM` variable systemd sets. Otherwise logs go to stdout,
colorized. The nix package pulls in `systemd-python`; elsewhere it is optional, and without it
the bot falls back to stdout.
