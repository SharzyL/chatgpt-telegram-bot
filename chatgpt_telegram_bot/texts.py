"""The bot's fixed wording: usage messages, the help page, the command menu and prompt guidance."""

from html import escape as html_escape

# Formatting guidance depends on which renderer the reply goes through, so it is appended to
# the default system prompt rather than baked into it. A custom prompt (`,[…]`) is left exactly
# as the user wrote it.
RICHTEXT_PROMPT_APPEND = (
    'Format the reply in Markdown; headings, tables, horizontal rules and LaTeX math all render.'
    + ' Delimit math with $...$ inline and $$...$$ on its own line.'
)
ENTITY_PROMPT_APPEND = (
    'Format the reply in Markdown, but do not use tables or LaTeX math: they do not render and'
    + ' would reach the user as raw source. Bold, italic, strikethrough, inline code, code blocks,'
    + ' links, blockquotes and lists all render.'
)


# Commands offered in the palette: only those that are safe to send bare, since picking one
# sends it at once. /export qualifies, its title being optional and the export harmless; a bare
# /regenerate would write over a reply before the user could add the argument they meant, and
# /manipulate or /import could only fail. Those are typed, and /help describes them.
BOT_COMMANDS = [
    ('ping', 'Test bot connectivity'),
    ('help', 'Show usage and commands'),
    ('models', 'List the models and their prefixes'),
    ('abort', 'Stop whatever is generating in this chat'),
    ('navigate', 'List the turns of the replied-to thread'),
    ('linearize', 'Delete every branch except the line to the replied message'),
    ('export', 'Save the replied-to thread as JSON'),
]

LINEARIZE_USAGE = """[!] `/linearize` prunes a thread, so it has to reply to a message in one.

It deletes every branch that grew from the same first message, keeping only the line down to the replied message — so anything after that message goes too. You confirm the list first."""

EXPORT_USAGE = """[!] `/export` saves a thread as JSON, so it has to reply to a message in one.

`/export` — titled after the first prompt
`/export <title>` — with that title

The file is in the format `/import` reads. Reply to a bot message to include its answer."""

NAVIGATE_USAGE = """[!] `/navigate` lists the turns of a thread, so it has to reply to a message in one.

Reply to a bot message to include its answer; the listing opens on the last page."""

REGENERATE_USAGE = """[!] `/regenerate` answers a prompt again in place, so it has to reply to a bot message:

`/regenerate` — the same prompt and model
`/regenerate <prefix>` — the same prompt, another model
`/regenerate <new prompt>` — another prompt, the same model
`/regenerate <prefix> <new prompt>` — both
`/regenerate <endpoint>/<model>[,flags] [new prompt]` — a model not in /models, as /chat names it

`/re` is short for `/regenerate`.

A first word that is a model prefix is read as one, so a new prompt that happens to start with a prefix needs the model named in front of it."""

IMPORT_USAGE = """[!] `/import` needs an exported conversation as a JSON file, either attached to the command or replied to.

`/import` — record no model; the first reply names one with a prefix
`/import <prefix>` — continue it with that model, overrides allowed

The file holds a `messages` array of `{role, content}` objects, alternating user and assistant."""

MANIPULATE_USAGE = """[!] `/manipulate` rewrites a bot reply, so it has to reply to one:

`/manipulate <text>`

or, to supply a chain of thought too:

`/manipulate <think>your cot</think> <text>`

`/mani` is short for `/manipulate`."""


NO_STORED_TEXT = '[!] No stored conversation for that message'

CHAT_USAGE = """[!] `/chat` starts a new conversation with the model you name:

`/chat <prefix>[,flags] <message>`

or, for a model not in /models:

`/chat <endpoint>/<model>[,suffix=<path>][,api_type=<type>][,flags] <message>`

Sent as a reply, it brings in the media of the message replied to but always starts afresh."""


def help_sections(system_prompt: str) -> list[str]:
    """The help page as HTML sections, which the bot packs into as few messages as fit."""
    usage = [
        '<b>Usage</b>  — <code>/models</code> lists the prefixes',
        '  <code>&lt;prefix&gt; &lt;message&gt;</code>  — start a conversation',
        '  <code>&lt;prefix&gt;,t &lt;message&gt;</code>  — enable thinking',
        '  <code>&lt;prefix&gt;,t=high &lt;message&gt;</code>  — set thinking effort',
        '  <code>&lt;prefix&gt;,t= &lt;message&gt;</code>  — disable thinking',
        '  <code>&lt;prefix&gt;,s &lt;message&gt;</code>  — enable search',
        '  <code>&lt;prefix&gt;,r &lt;message&gt;</code>  — let Telegram render the markdown',
        '  <code>&lt;prefix&gt;,r= &lt;message&gt;</code>  — render to entities instead',
        '  <code>&lt;prefix&gt;,dt &lt;message&gt;</code>  — dummy tool: pass earlier thinking back (DeepSeek)',
        '  <code>&lt;prefix&gt;,[custom prompt] &lt;message&gt;</code>  — custom system prompt',
        '  <code>&lt;prefix&gt;,+[extra prompt] &lt;message&gt;</code>  — append to default prompt',
        '  <code>&lt;prefix&gt;,[] &lt;message&gt;</code>  — clear system prompt',
        '  <code>&lt;prefix&gt;,suffix=/v1 &lt;message&gt;</code>  — another path on the endpoint',
        '  <code>&lt;prefix&gt;,api_type=openai_legacy &lt;message&gt;</code>  — another API'
        + ' (<code>openai</code>, <code>openai_legacy</code>, <code>anthropic</code>)',
        'Reply to a bot message to continue; reply to an older one to fork.',
        'Reply to a photo or file with a prefix to talk about it.',
        'A prefix on a reply switches the model from there on.',
    ]

    commands = [
        '<b>Commands</b>',
        '  <code>/help</code>  — this page',
        '  <code>/models</code>  — the models and their prefixes',
        '  <code>/ping</code>  — chat id, and whether the bot answers here',
        '  <code>/abort</code>  — stop whatever is generating in this chat',
        'Sent as a reply to a message in a conversation:',
        '  <code>/navigate</code>  — list its turns, one line each, paged',
        '  <code>/linearize</code>  — delete every other branch, after confirming',
        '  <code>/export [title]</code>  — save it as JSON that <code>/import</code> reads',
        'Typed, not in the command menu, as they act on their arguments:',
        '  <code>/chat &lt;prefix&gt;[,flags] &lt;message&gt;</code>',
        '  <code>/chat &lt;endpoint&gt;/&lt;model&gt;[,flags] &lt;message&gt;</code>',
        '    Start a new conversation, with any model an endpoint serves, configured or not.'
        + " As a reply it takes the media replied to, a bot reply's included, and never continues a thread.",
        '  <code>/regenerate [prefix] [new prompt]</code>, or <code>/re</code>',
        '    Reply to a bot message to answer its prompt again, over the old reply.'
        + ' A prefix, or <code>&lt;endpoint&gt;/&lt;model&gt;</code> as in <code>/chat</code>,'
        + ' switches the model and new text replaces the prompt;'
        + ' a first word that is either is read as one.',
        '  <code>/manipulate &lt;text&gt;</code>, or <code>/mani</code>',
        '    Reply to a bot message to rewrite what it said; later turns build on yours.'
        + ' Lead with <code>&lt;think&gt;…&lt;/think&gt;</code> to supply its reasoning, sent under dt.',
        '  <code>/import [prefix]</code>',
        '    Attach an exported JSON file, or reply to one, to replay it as messages to reply to.'
        + ' Without a prefix, the first reply chooses the model.',
    ]

    prompt = [
        f'<b>Default system prompt</b>\n<code>{html_escape(system_prompt)}</code>',
        '',
        'Formatting guidance is appended to it, depending on <code>richtext</code>:',
        f'  on → <code>{html_escape(RICHTEXT_PROMPT_APPEND)}</code>',
        f'  off → <code>{html_escape(ENTITY_PROMPT_APPEND)}</code>',
    ]
    return ['\n'.join(section) for section in (usage, commands, prompt)]
