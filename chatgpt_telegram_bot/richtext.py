from __future__ import annotations

from typing import Any, override

import mistune
from telethon import types

_md_parser = mistune.create_markdown(renderer='ast', plugins=['strikethrough'])


class RichText:
    def __init__(self, s: str | RichText | list[dict[str, Any]]) -> None:
        if isinstance(s, str):
            self.children: list[dict[str, Any]] = [{'type': 'text', 'content': s}]
        elif isinstance(s, RichText):
            self.children = s.children
        elif (
            isinstance(s, list)  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime validation
            and s
            and all(isinstance(c, dict) for c in s)  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime validation
            and all('type' in c and 'content' in c for c in s)
        ):
            self.children = s
        else:
            raise ValueError()

    @classmethod
    def Raw(cls, s: str) -> RichText:
        return RichText([{'type': 'text', 'content': s}])

    @classmethod
    def Bold(cls, s: str | RichText) -> RichText:
        return RichText([{'type': 'bold', 'content': RichText(s)}])

    @classmethod
    def Italic(cls, s: str | RichText) -> RichText:
        return RichText([{'type': 'italic', 'content': RichText(s)}])

    @classmethod
    def Strikethrough(cls, s: str | RichText) -> RichText:
        return RichText([{'type': 'strikethrough', 'content': RichText(s)}])

    @classmethod
    def Code(cls, s: str) -> RichText:
        return RichText([{'type': 'code', 'content': s}])

    @classmethod
    def Pre(cls, s: str, language: str = '') -> RichText:
        return RichText([{'type': 'pre', 'content': s, 'language': language}])

    @classmethod
    def Href(cls, s: str | RichText, url: str) -> RichText:
        return RichText([{'type': 'href', 'content': RichText(s), 'url': url}])

    @classmethod
    def Blockquote(cls, s: str | RichText, collapsed: bool = False) -> RichText:
        return RichText([{'type': 'blockquote', 'content': RichText(s), 'collapsed': collapsed}])

    def collapse_first_blockquote(self) -> None:
        """
        Mark the leading quote foldable, which for a formatted reply is the thinking block.

        Only the entity a client is given carries this; nothing about a quote's content
        makes it fold on its own.
        """
        for child in self.children:
            if child['type'] == 'blockquote':
                child['collapsed'] = True
                return

    def __len__(self) -> int:
        return sum(len(c['content']) for c in self.children)

    @override
    def __str__(self) -> str:
        return f'RichText({self.children})'

    @override
    def __repr__(self) -> str:
        return f'RichText({self.children})'

    def __add__(self, value: RichText | str) -> RichText:
        if isinstance(value, RichText):
            if len(self) == 0:
                return value
            if len(value) == 0:
                return self
            self_last = self.children[-1].copy()
            value_first = value.children[0].copy()
            self_last_content = self_last['content']
            value_first_content = value_first['content']
            del self_last['content']
            del value_first['content']
            if self_last == value_first:
                self_last['content'] = self_last_content + value_first_content
                return RichText(self.children[:-1] + [self_last] + value.children[1:])
            return RichText(self.children + value.children)
        else:
            return self + RichText(value)

    def __radd__(self, value: str) -> RichText:
        return RichText(value) + self

    @override
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, RichText):
            if isinstance(value, str):
                return self == RichText(value)
            return False
        return self.children == value.children

    def __getitem__(self, key: slice) -> RichText:
        if not isinstance(key, slice):  # pyright: ignore[reportUnnecessaryIsInstance]  # runtime guard
            raise NotImplementedError()  # pyright: ignore[reportUnreachable]
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise NotImplementedError()
        if start >= stop:
            return RichText('')
        offset = 0
        new_children = []
        for c in self.children:
            clen = len(c['content'])
            c_start = offset
            c_stop = c_start + clen
            i_start = max(c_start, start)
            i_stop = min(c_stop, stop)
            if i_start < i_stop:
                new_c = c.copy()
                new_c['content'] = c['content'][i_start - offset : i_stop - offset]
                new_children.append(new_c)
            offset += clen
        return RichText(new_children)

    @classmethod
    def from_markdown(cls, markdown: str) -> RichText:
        ast: list[dict[str, Any]] = _md_parser(markdown)  # pyright: ignore[reportAssignmentType]  # ast renderer always returns list
        return _render_blocks(ast)

    def to_telegram(self, offset: int = 0) -> tuple[str, list[Any]]:
        def utf16len(s: str) -> int:
            return len(s.encode('utf-16-le')) // 2

        def strip_entity(s: str) -> tuple[int, int]:
            lstripped = s[: len(s) - len(s.lstrip())]
            return utf16len(lstripped), utf16len(s.strip())

        entities = []
        text = ''
        for c in self.children:
            if c['type'] == 'text':
                text += c['content']
                offset += utf16len(c['content'])
            elif c['type'] == 'bold':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                start, length = strip_entity(t)
                if length:
                    entities.append(types.MessageEntityBold(offset + start, length))
                offset += utf16len(t)
            elif c['type'] == 'italic':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                start, length = strip_entity(t)
                if length:
                    entities.append(types.MessageEntityItalic(offset + start, length))
                offset += utf16len(t)
            elif c['type'] == 'strikethrough':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                start, length = strip_entity(t)
                if length:
                    entities.append(types.MessageEntityStrike(offset + start, length))
                offset += utf16len(t)
            elif c['type'] == 'code':
                text += c['content']
                start, length = strip_entity(c['content'])
                if length:
                    entities.append(types.MessageEntityCode(offset + start, length))
                offset += utf16len(c['content'])
            elif c['type'] == 'pre':
                text += c['content']
                start, length = strip_entity(c['content'])
                if length:
                    entities.append(types.MessageEntityPre(offset + start, length, c['language']))
                offset += utf16len(c['content'])
            elif c['type'] == 'href':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                start, length = strip_entity(t)
                if length:
                    entities.append(types.MessageEntityTextUrl(offset + start, length, c['url']))
                offset += utf16len(t)
            elif c['type'] == 'blockquote':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                t_len = utf16len(t)
                if t_len:
                    # `collapsed` is what makes a quote foldable in the clients; it is left
                    # unset rather than False so the flag is simply absent when not wanted
                    collapsed = c.get('collapsed') or None
                    entities.append(types.MessageEntityBlockquote(offset, t_len, collapsed=collapsed))
                offset += t_len
        return text, entities


def _render_node(node: dict[str, Any]) -> RichText:
    t = node['type']
    if t == 'text':
        return RichText(node['raw'])
    elif t in ('paragraph', 'block_text'):
        return _render_children(node)
    elif t == 'heading':
        level = (node.get('attrs') or {}).get('level', 1)
        return RichText.Bold('#' * level + ' ' + _render_children(node))
    elif t == 'strong':
        return RichText.Bold(_render_children(node))
    elif t == 'emphasis':
        return RichText.Italic(_render_children(node))
    elif t == 'strikethrough':
        return RichText.Strikethrough(_render_children(node))
    elif t == 'codespan':
        return RichText.Code(node['raw'])
    elif t == 'block_code':
        info = (node.get('attrs') or {}).get('info', '')
        raw = node.get('raw', '')
        raw = raw.removesuffix('\n')
        return RichText.Pre(raw, info)
    elif t == 'link':
        url = (node.get('attrs') or {}).get('url', '')
        return RichText.Href(_render_children(node), url)
    elif t == 'image':
        url = (node.get('attrs') or {}).get('url', '')
        children = _render_children(node)
        return RichText.Href(children if len(children) > 0 else RichText(url), url)
    elif t == 'block_quote':
        return RichText.Blockquote(_render_blocks(node.get('children') or []))
    elif t == 'list':
        return _render_list(node)
    elif t in ('softbreak', 'hardbreak'):
        return RichText('\n')
    elif t == 'thematic_break':
        return RichText('---')
    elif t == 'blank_line':
        return RichText('')
    else:
        if 'raw' in node:
            return RichText(node['raw'])
        if node.get('children'):
            return _render_children(node)
        return RichText('')


def _render_children(node: dict[str, Any]) -> RichText:
    result = RichText('')
    for child in node.get('children') or []:
        result = result + _render_node(child)
    return result


def _render_blocks(nodes: list[dict[str, Any]]) -> RichText:
    result = RichText('')
    first = True
    for node in nodes:
        if node['type'] == 'blank_line':
            continue
        if not first:
            result = result + '\n\n'
        first = False
        result = result + _render_node(node)
    return result


def _render_list(node: dict[str, Any]) -> RichText:
    ordered = (node.get('attrs') or {}).get('ordered', False)
    result = RichText('')
    for i, item in enumerate(node.get('children') or []):
        if i > 0:
            result = result + '\n'
        marker = f'{i + 1}. ' if ordered else '• '
        result = result + marker + _render_blocks(item.get('children') or [])
    return result
