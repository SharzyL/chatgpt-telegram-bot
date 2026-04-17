from typing import Any

from chatgpt_telegram_bot.richtext import RichText


def format_reply(
    thinking: str,
    reply: str,
    thinking_done: bool,
    status: str | None = None,
    tool_calls: list[str] | None = None,
    usage: dict[str, Any] | None = None,
) -> str | RichText:
    suffix = f' [!{status}]' if status else ''
    if not thinking_done:
        # still in thinking phase, or no thinking at all
        if not thinking:
            return suffix.strip() if suffix else ''
        return RichText.Blockquote(RichText.from_markdown(thinking) + suffix)
    # thinking is done, show blockquote + response
    result: str | RichText = ''
    if thinking:
        result = RichText.Blockquote(RichText.from_markdown(thinking.rstrip('\n')))
    if reply or suffix:
        sep = '\n\n' if thinking else ''
        result = result + sep + RichText.from_markdown(reply) + suffix
    if tool_calls or usage:
        footer_content: RichText | str = ''
        if tool_calls:
            footer_content = footer_content + RichText.Bold('Tool calls') + '\n'
            footer_content = footer_content + '\n'.join(f'🔍 {q}' for q in tool_calls)
        if usage:
            if footer_content:
                footer_content = footer_content + '\n\n'
            footer_content = footer_content + RichText.Bold('Usage') + '\n'
            footer_content = footer_content + ', '.join(f'{k}={v}' for k, v in usage.items())
        result = result + '\n\n' + RichText.Blockquote(footer_content)
    return result
