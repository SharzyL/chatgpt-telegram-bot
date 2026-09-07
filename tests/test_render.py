"""Tests for the two renderers: how a reply is cut and which request fields it fills."""

from __future__ import annotations

from typing import Any

from chatgpt_telegram_bot.render import (
    ENTITY_RENDERER,
    PLAIN_LENGTH_LIMIT,
    SERVER_MARKDOWN_RENDERER,
    Rendered,
)
from chatgpt_telegram_bot.reply import format_reply
from chatgpt_telegram_bot.utils import telegram_len

PREFIX = '🤖 `gpt`\n\n'


class TestEntityRenderer:
    def test_markdown_becomes_entities(self):
        bodies = ENTITY_RENDERER.render('**bold** and `code`')
        assert len(bodies) == 1
        assert bodies[0].text == 'bold and code'
        assert [(type(e).__name__, e.offset, e.length) for e in bodies[0].entities] == [
            ('MessageEntityBold', 0, 4),
            ('MessageEntityCode', 9, 4),
        ]

    def test_no_rich_markdown_is_sent(self):
        assert ENTITY_RENDERER.render('hi')[0].rich_markdown is None

    def test_blockquote_survives(self):
        bodies = ENTITY_RENDERER.render('>quoted')
        assert bodies[0].text == 'quoted'
        assert [type(e).__name__ for e in bodies[0].entities] == ['MessageEntityBlockquote']

    def test_prefix_offsets_are_utf16(self):
        # the robot emoji is two UTF-16 units, so the code entity cannot start at index 2
        bodies = ENTITY_RENDERER.render('hi', PREFIX)
        code = next(e for e in bodies[0].entities if type(e).__name__ == 'MessageEntityCode')
        assert code.offset == 3
        assert bodies[0].text.startswith('🤖 gpt')

    def test_prefix_only_on_first_body(self):
        bodies = ENTITY_RENDERER.render('x' * 9000, PREFIX)
        assert len(bodies) > 1
        assert bodies[0].text.startswith('🤖 gpt')
        assert not bodies[1].text.startswith('🤖')

    def test_no_body_exceeds_the_limit(self):
        bodies = ENTITY_RENDERER.render('y' * 9000, PREFIX)
        assert all(telegram_len(b.text) <= PLAIN_LENGTH_LIMIT for b in bodies)

    def test_astral_characters_do_not_overflow(self):
        # 3000 emoji are 3000 characters but 6000 UTF-16 units
        bodies = ENTITY_RENDERER.render('😀' * 3000)
        assert all(telegram_len(b.text) <= PLAIN_LENGTH_LIMIT for b in bodies)
        assert len(bodies) > 1

    def test_nothing_is_lost_across_a_split(self):
        bodies = ENTITY_RENDERER.render('z' * 9000)
        assert ''.join(b.text for b in bodies) == 'z' * 9000

    def test_empty_reply_still_has_a_body(self):
        assert ENTITY_RENDERER.render('') == [Rendered(text='')]

    def test_whitespace_only_reply_still_has_a_body(self):
        assert ENTITY_RENDERER.render('   \n\n  ') == [Rendered(text='')]

    def test_astral_text_fills_its_bodies(self):
        """A UTF-16 budget spent one character at a time would emit thousands of messages."""
        bodies = ENTITY_RENDERER.render('😀' * 6000)
        assert len(bodies) == 3
        assert telegram_len(bodies[0].text) > PLAIN_LENGTH_LIMIT - 4

    def test_bodies_carry_no_edge_whitespace(self):
        # Telegram strips it, which would shift every entity in the body left
        bodies = ENTITY_RENDERER.render('para text here\n\n' * 900, PREFIX)
        assert len(bodies) > 1
        assert all(b.text == b.text.strip() for b in bodies)

    def test_entities_stay_in_bounds_after_telegram_trims(self):
        source = format_reply('Reasoning step. ' * 900, 'Answer **now** with `code`.', True)
        bodies = ENTITY_RENDERER.render(source, PREFIX)
        assert len(bodies) > 1
        for body in bodies:
            trimmed = telegram_len(body.text.strip())
            assert all(e.length > 0 and e.offset >= 0 for e in body.entities)
            assert all(e.offset + e.length <= trimmed for e in body.entities)


class TestServerMarkdownRenderer:
    def test_source_is_passed_through(self):
        bodies = SERVER_MARKDOWN_RENDERER.render('# Head\n\n| a | b |')
        assert len(bodies) == 1
        assert bodies[0].rich_markdown == '# Head\n\n| a | b |'
        assert bodies[0].entities == []

    def test_message_field_is_a_truncated_copy(self):
        bodies = SERVER_MARKDOWN_RENDERER.render('para\n\n' * 3000)
        assert telegram_len(bodies[0].text) <= PLAIN_LENGTH_LIMIT
        assert len(bodies[0].rich_markdown or '') > PLAIN_LENGTH_LIMIT

    def test_prefix_lands_on_the_first_body_only(self):
        bodies = SERVER_MARKDOWN_RENDERER.render('body', PREFIX)
        assert bodies[0].rich_markdown == PREFIX + 'body'

    def test_long_reply_splits(self):
        bodies = SERVER_MARKDOWN_RENDERER.render('para\n\n' * 9000, PREFIX)
        assert len(bodies) > 1
        assert (bodies[0].rich_markdown or '').startswith(PREFIX)
        assert not (bodies[1].rich_markdown or '').startswith(PREFIX)

    def test_no_body_exceeds_the_rich_limit(self):
        bodies = SERVER_MARKDOWN_RENDERER.render('para\n\n' * 9000)
        assert all(telegram_len(b.rich_markdown or '') <= SERVER_MARKDOWN_RENDERER.limit for b in bodies)

    def test_empty_reply_still_has_a_body(self):
        assert SERVER_MARKDOWN_RENDERER.render('') == [Rendered(text='', rich_markdown='')]


class TestRenderedEquality:
    """BotReplyMessages skips an edit when a body is unchanged, so equality has to hold."""

    def test_same_input_compares_equal(self):
        assert ENTITY_RENDERER.render('**a**', PREFIX) == ENTITY_RENDERER.render('**a**', PREFIX)
        assert SERVER_MARKDOWN_RENDERER.render('a', PREFIX) == SERVER_MARKDOWN_RENDERER.render('a', PREFIX)

    def test_changed_text_compares_unequal(self):
        assert ENTITY_RENDERER.render('**a**') != ENTITY_RENDERER.render('**b**')

    def test_changed_markup_alone_compares_unequal(self):
        # same visible text, different entities
        assert ENTITY_RENDERER.render('**a**') != ENTITY_RENDERER.render('a')


class TestImageNeutralization:
    """Telegram rejects a rich message whose markdown holds an image with no media."""

    def test_inline_image_becomes_a_link(self):
        body = SERVER_MARKDOWN_RENDERER.render('see ![a cat](http://x/c.png) here')[0]
        assert body.rich_markdown == 'see [a cat](http://x/c.png) here'

    def test_reference_image_becomes_a_link(self):
        body = SERVER_MARKDOWN_RENDERER.render('![alt][ref]')[0]
        assert body.rich_markdown == '[alt][ref]'

    def test_empty_alt_uses_the_target_as_text(self):
        body = SERVER_MARKDOWN_RENDERER.render('![](http://x/c.png)')[0]
        assert body.rich_markdown == '[http://x/c.png](http://x/c.png)'

    def test_several_images_are_all_demoted(self):
        body = SERVER_MARKDOWN_RENDERER.render('![a](1.png) and ![b](2.png)')[0]
        assert body.rich_markdown == '[a](1.png) and [b](2.png)'

    def test_image_inside_a_code_span_is_left_alone(self):
        source = 'use `![alt](url)` for images'
        assert SERVER_MARKDOWN_RENDERER.render(source)[0].rich_markdown == source

    def test_image_inside_a_fence_is_left_alone(self):
        source = '```md\n![alt](url)\n```'
        assert SERVER_MARKDOWN_RENDERER.render(source)[0].rich_markdown == source

    def test_plain_link_is_untouched(self):
        source = '[not an image](http://x)'
        assert SERVER_MARKDOWN_RENDERER.render(source)[0].rich_markdown == source

    def test_bang_without_a_link_is_untouched(self):
        source = 'wow! [link](http://x) and ![incomplete'
        assert SERVER_MARKDOWN_RENDERER.render(source)[0].rich_markdown == source

    def test_entity_mode_renders_the_image_as_a_link(self):
        # nothing to neutralize here: an image has no entity, so RichText makes it a link
        body = ENTITY_RENDERER.render('![a cat](http://x/c.png)')[0]
        assert body.text == 'a cat'
        assert [type(e).__name__ for e in body.entities] == ['MessageEntityTextUrl']


class TestCollapsibleQuotes:
    """A quote folds in the clients only when its entity carries `collapsed`."""

    @staticmethod
    def quotes(body: Rendered) -> list[Any]:
        return [e for e in body.entities if type(e).__name__ == 'MessageEntityBlockquote']

    def test_thinking_quote_is_collapsed_however_short(self):
        source = format_reply('Brief.', 'Answer.', True)
        assert [e.collapsed for e in self.quotes(ENTITY_RENDERER.render(source, PREFIX, True)[0])] == [True]

    def test_footer_stays_open_while_thinking_folds(self):
        source = format_reply('Reasoning at length. ' * 40, 'The answer.', True, usage={'in': 84})
        thinking, footer = self.quotes(ENTITY_RENDERER.render(source, PREFIX, True)[0])
        assert thinking.collapsed is True
        assert footer.collapsed is None

    def test_model_quotes_stay_open_without_thinking(self):
        source = format_reply('', '> a quote the model wrote\n\nanswer', True, usage={'in': 1})
        found = self.quotes(ENTITY_RENDERER.render(source, PREFIX, False)[0])
        assert len(found) == 2
        assert all(e.collapsed is None for e in found)

    def test_only_the_leading_quote_folds(self):
        source = format_reply('Thinking.', '> the model also quotes\n\nanswer', True)
        found = self.quotes(ENTITY_RENDERER.render(source, PREFIX, True)[0])
        assert [e.collapsed for e in found] == [True, None]

    def test_a_collapsed_quote_split_across_bodies_stays_collapsed(self):
        bodies = ENTITY_RENDERER.render(format_reply('Reasoning step. ' * 900, 'Answer.', True), PREFIX, True)
        assert len(bodies) > 1
        assert all(e.collapsed is True for b in bodies for e in self.quotes(b))
