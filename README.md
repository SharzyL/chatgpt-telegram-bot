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
- **Regeneration** — `/regenerate` answers a prompt again in place, optionally on another
  model or with the prompt reworded.
- **Conversation import** — `/import` replays a conversation exported from elsewhere into the
  chat, one message per turn, each of them replyable.
- **Long replies** — output over Telegram's per-message limit is split across several messages.
- **Multimodal input** — photos and image files (single or albums) and UTF-8 text files,
  attached or in a message you reply to.
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

In a group the bot answers two things and stays silent otherwise: a message starting with a
model prefix, and a reply to one of its own messages. Forum topics are supported, including
topics other than General.

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
| `gpt,dt hello` | dummy tool: pass earlier chains of thought back, the way DeepSeek needs (see below) |
| `gpt,dt= hello` | no dummy tool, and earlier chains of thought dropped (the default) |
| `gpt,live hello` | live: editing any prompt of this conversation answers it again, over its reply; replies get Regenerate and Delete buttons |
| `gpt,[be terse] hello` | replace the system prompt for this conversation |
| `gpt,+[be terse] hello` | append to the default system prompt |
| `gpt,[] hello` | send no system prompt at all |
| `gpt,suffix=/v2 hello` | call another path on the endpoint (`suffix=` for none) |
| `gpt,api_type=openai_legacy hello` | use another backend: `openai`, `openai_legacy` or `anthropic` |

Several can be combined (`gpt,t=high,s hello`), and brackets may contain spaces and escaped
brackets (`\[`, `\]`).

In a `live` conversation, editing a prompt you sent answers it again: the edited text is read
as if you had sent it fresh (a prefix in it can switch the model), and the new answer is written
over the old reply, its header showing the next `version=`. This works for any prompt of the
conversation, not only the first. A conversation has to be live already for an edit to count,
so adding `,live` to a prompt by editing it does not make that edit regenerate. An answer still
streaming when you edit is stopped, and the latest edit wins.

Each reply in a live conversation carries two buttons, from its first streamed message on. **Regenerate** answers
its prompt again, as `/regenerate` does. **Delete** removes every message of that reply, leaving
your prompt (edit it to get a new answer); it is refused once the conversation has gone on from
that reply, since a later turn would lose what it answered. Pressed while the reply is still
being written, either button stops it first: Regenerate then starts the answer over.

A model's chain of thought is shown with each reply, but by default earlier ones are not sent
back. With `dt` (dummy tool) they are, as reasoning: DeepSeek keeps a passed-back chain of
thought only in a request that offers a tool, so `dt` adds one that does nothing (about 230
extra prompt tokens a request, almost all of them DeepSeek's own tool prompt) and sends the earlier chains of thought with it. It is meant
for DeepSeek models on the Responses API (`api_type = "openai"`, the default) and refused for
other backends; OpenAI models carry their own encrypted reasoning without it. A chain of thought supplied
with `/manipulate` also reaches the model only under `dt`.

`/chat` starts a conversation with a model you name, even one that is not in `bot.toml`:

```
/chat gpt,t=high hello
/chat openrouter/vendor/some-model,suffix=/api/v1,api_type=openai_legacy hello
```

The first word is either a prefix, read with its overrides as usual, or `<endpoint>/<model>`:
an endpoint from `bot.toml` (`/models` lists them) and the model name sent to it, which may
itself contain `/`. A configured prefix wins, so a prefix spelt like a spec, say `o/gf`, means
the model it is configured as. The model and its flags are remembered for the conversation,
so a plain reply continues on it. `/chat` always starts a new conversation, also when sent as a
reply: the photos and files of the message it replies to come along, even those of a bot reply,
but the thread that message belongs to does not. That is the way to talk about media
the bot sent in a reply, since a plain reply to it would just continue its thread.

Commands (picking one from Telegram's command menu sends it at once, so the menu holds only
those safe to send bare; type `/chat`, `/regenerate`, `/manipulate` and `/import`, and see `/help` for
all of them):

- `/help` — the override syntax, the commands and the current system prompt
- `/models` — every model prefix and the model behind it, and the endpoints `/chat` can name
- `/ping` — chat id, user id, and whether the chat is allowed
- `/chat <prefix>[,flags] <message>`, or `/chat <endpoint>/<model>[,flags] <message>` — start
  a new conversation with the model named, configured or not (see above)
- `/manipulate <text>`, or `/manipulate <think>reasoning</think> <text>` — reply to a bot
  message with this to rewrite what it is recorded as having said. `/mani` is short for
  `/manipulate`
- `/import`, or `/import <prefix>` — attach a JSON export (or reply to one) to replay a
  conversation into the chat
- `/regenerate`, `/regenerate <prefix>`, `/regenerate <new prompt>`,
  `/regenerate <prefix> <new prompt>` — reply to a bot message to answer its prompt again,
  writing over the reply in place. A first word that is a model prefix is read as one, and so
  is `<endpoint>/<model>[,flags]` on an endpoint from `bot.toml`, as `/chat` takes it.
  A regenerated reply's header shows which version of the answer it is (`version=2` for the
  first regeneration, the original being version 1). `/re` is short for `/regenerate`
- `/abort` — stop whatever is generating in this chat; the reply keeps what it had, marked
  `[!aborted]`, and can still be replied to
- `/navigate` — reply to a message to list the turns of its thread, one line each, paged with
  buttons; in a supergroup each answer links to its message
- `/export`, or `/export <title>` — reply to a message to save its thread as JSON in the format
  `/import` reads
- `/linearize` — reply to a message to delete every other branch of its thread, keeping the line
  down to it; lists what will go and asks you to confirm. In a group the bot needs delete
  rights to remove other people's messages

Attach a photo, an image file or a UTF-8 text file to a message, with or without a caption.
When sending them together is awkward, send or forward the media first and then reply to it
with a prefixed message: the photos and files of the message replied to — all of an album —
become part of the prompt (its caption does not), and a new conversation starts from there. This
works for files the bot sent too, such as an `/export`. `/import` likewise reads a file it
replies to.

`/import` takes a conversation exported as JSON:

```json
{
  "title": "Reading list",
  "messages": [
    { "role": "user", "content": "what should I read?" },
    { "role": "assistant", "content": "start with ...", "reasoning": "optional chain of thought" }
  ]
}
```

Roles must alternate, starting with the user. An answer may carry its chain of thought in an
optional `reasoning` field, which `/export` writes for answers that have one. The bot confirms
the title and then posts one message per turn: the prompt, a `-----` line, then the answer,
with its chain of thought folded above it like a live reply's. Reply to any of them to continue
the conversation from that point; under `dt`, the imported chains of thought go back to the
model too.

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
| `db_compression` | `true` | compress stored conversations with zstd (needs Python 3.14+) |

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
| `dummy_tool` | `false` | offer a do-nothing tool and pass earlier chains of thought back (`,dt`); `api_type = "openai"` only |
| `live` | `false` | editing a prompt answers it again over its reply (`,live`) |
| `system_prompt` | `null` | replaces the default prompt for this model |
| `system_prompt_append` | `null` | appended to the default prompt |
| `no_system_prompt` | `false` | send no system prompt |

Persistent state — the conversation database, the photo cache and the Telethon session — goes
to `$XDG_STATE_HOME/chatgpt-telegram-bot` (`~/.local/state/…` by default), or wherever
`--data-dir` points.

The conversation database is `db-v2.sqlite3`. Versions before it kept conversations in a `shelve`
file named `db`; on the first start with that file and no `db-v2.sqlite3`, the bot copies every
conversation across, checks that each one reads back unchanged, and only then starts. The old
`db` is left in place as a backup and can be deleted once you are satisfied.

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
