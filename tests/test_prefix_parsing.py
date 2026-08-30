"""Tests for prefix parsing: parse_overrides, find_delimiter_outside_brackets, and match_prefix."""

from __future__ import annotations

from chatgpt_telegram_bot.utils import (
    find_delimiter_outside_brackets,
    match_prefix,
    parse_overrides,
    split_respecting_brackets,
)

# ---------------------------------------------------------------------------
# split_respecting_brackets
# ---------------------------------------------------------------------------


class TestSplitRespectingBrackets:
    def test_simple_comma_split(self):
        assert split_respecting_brackets('a,b,c') == ['a', 'b', 'c']

    def test_empty_string(self):
        assert split_respecting_brackets('') == []

    def test_no_delimiter(self):
        assert split_respecting_brackets('abc') == ['abc']

    def test_brackets_protect_commas(self):
        assert split_respecting_brackets('a,[hello, world],b') == ['a', '[hello, world]', 'b']

    def test_escaped_brackets_dont_nest(self):
        assert split_respecting_brackets(r'a,[\[inner\]],b') == ['a', r'[\[inner\]]', 'b']

    def test_nested_brackets(self):
        assert split_respecting_brackets('a,[outer [inner, x]],b') == ['a', '[outer [inner, x]]', 'b']

    def test_trailing_comma(self):
        assert split_respecting_brackets('a,b,') == ['a', 'b']

    def test_leading_comma(self):
        assert split_respecting_brackets(',a') == ['', 'a']

    def test_bracket_only(self):
        assert split_respecting_brackets('[a,b,c]') == ['[a,b,c]']


# ---------------------------------------------------------------------------
# find_delimiter_outside_brackets
# ---------------------------------------------------------------------------


class TestFindDelimiterOutsideBrackets:
    def test_space_delimiter(self):
        assert find_delimiter_outside_brackets('t=high hello') == 6

    def test_dollar_not_a_delimiter(self):
        assert find_delimiter_outside_brackets('t=high$hello') == -1

    def test_no_delimiter(self):
        assert find_delimiter_outside_brackets('t=high') == -1

    def test_space_inside_brackets_skipped(self):
        assert find_delimiter_outside_brackets('[hello world] text') == 13

    def test_dollar_inside_brackets_not_a_delimiter(self):
        assert find_delimiter_outside_brackets('[cost $5]$text') == -1

    def test_escaped_bracket_not_counted(self):
        # \[ does not open a bracket group, so the space after it is a delimiter
        assert find_delimiter_outside_brackets(r'\[ hello') == 2

    def test_brackets_with_overrides(self):
        assert find_delimiter_outside_brackets('t=high,[You are a bot] hello') == 22

    def test_empty_string(self):
        assert find_delimiter_outside_brackets('') == -1

    def test_space_at_start(self):
        assert find_delimiter_outside_brackets(' hello') == 0


# ---------------------------------------------------------------------------
# parse_overrides
# ---------------------------------------------------------------------------


class TestParseOverrides:
    def test_empty_string(self):
        assert parse_overrides('') == {}

    def test_bare_key(self):
        assert parse_overrides('t') == {'t': None}

    def test_key_value(self):
        assert parse_overrides('t=high') == {'t': 'high'}

    def test_multiple_overrides(self):
        assert parse_overrides('t=high,s') == {'t': 'high', 's': None}

    def test_system_prompt_brackets(self):
        assert parse_overrides('[You are a bot]') == {'system_prompt': 'You are a bot'}

    def test_system_prompt_with_commas(self):
        result = parse_overrides('[Hello, world, test]')
        assert result == {'system_prompt': 'Hello, world, test'}

    def test_system_prompt_with_other_overrides(self):
        result = parse_overrides('t=high,[You are a bot],s')
        assert result == {'t': 'high', 'system_prompt': 'You are a bot', 's': None}

    def test_system_prompt_escaped_brackets(self):
        result = parse_overrides(r'[Respond with \[1,2,3\]]')
        assert result == {'system_prompt': 'Respond with [1,2,3]'}

    def test_system_prompt_with_template_vars(self):
        result = parse_overrides('[Current time {current_time}, model {model}]')
        assert result == {'system_prompt': 'Current time {current_time}, model {model}'}

    def test_whitespace_stripped(self):
        assert parse_overrides(' t = high , s ') == {'t': 'high', 's': None}

    def test_empty_value(self):
        assert parse_overrides('t=') == {'t': ''}

    def test_value_with_equals(self):
        assert parse_overrides('t=a=b') == {'t': 'a=b'}

    def test_system_prompt_append(self):
        assert parse_overrides('+[Reply in JSON]') == {'system_prompt_append': 'Reply in JSON'}

    def test_system_prompt_append_with_commas(self):
        result = parse_overrides('+[Be concise, clear]')
        assert result == {'system_prompt_append': 'Be concise, clear'}

    def test_system_prompt_append_with_overrides(self):
        result = parse_overrides('t=high,+[Reply in JSON]')
        assert result == {'t': 'high', 'system_prompt_append': 'Reply in JSON'}

    def test_system_prompt_append_escaped_brackets(self):
        result = parse_overrides(r'+[Use format \[key: value\]]')
        assert result == {'system_prompt_append': 'Use format [key: value]'}


# ---------------------------------------------------------------------------
# ChatGPTTelegramBot.match_prefix
# ---------------------------------------------------------------------------


class TestMatchPrefix:
    # --- basic prefix matching ---

    def test_bare_prefix(self):
        assert match_prefix('om', 'om') == ('', {})

    def test_prefix_space_text(self):
        assert match_prefix('om hello world', 'om') == ('hello world', {})

    def test_prefix_dollar_no_match(self):
        assert match_prefix('om$hello world', 'om') is None

    def test_no_match(self):
        assert match_prefix('xx hello', 'om') is None

    def test_prefix_is_substring(self):
        assert match_prefix('omega hello', 'om') is None

    # --- overrides ---

    def test_overrides_bare_key(self):
        assert match_prefix('om,t hello', 'om') == ('hello', {'t': None})

    def test_overrides_key_value(self):
        assert match_prefix('om,t=high hello', 'om') == ('hello', {'t': 'high'})

    def test_overrides_multiple(self):
        assert match_prefix('om,t=high,s hello', 'om') == ('hello', {'t': 'high', 's': None})

    def test_overrides_dollar_not_a_delim(self):
        assert match_prefix('om,t=high$hello', 'om') == ('', {'t': 'high$hello'})

    def test_overrides_empty_value(self):
        assert match_prefix('om,t= hello', 'om') == ('hello', {'t': ''})

    def test_overrides_empty_value_no_text(self):
        assert match_prefix('om,t=', 'om') == ('', {'t': ''})

    def test_overrides_no_text(self):
        assert match_prefix('om,t=high', 'om') == ('', {'t': 'high'})

    # --- inline system prompt ---

    def test_system_prompt_basic(self):
        result = match_prefix('om,[You are a bot] hello', 'om')
        assert result is not None
        assert result == ('hello', {'system_prompt': 'You are a bot'})

    def test_system_prompt_with_spaces(self):
        result = match_prefix('om,[You are a helpful assistant] hello world', 'om')
        assert result is not None
        assert result == ('hello world', {'system_prompt': 'You are a helpful assistant'})

    def test_system_prompt_with_commas(self):
        result = match_prefix('om,[Be concise, clear, and helpful] hello', 'om')
        assert result is not None
        assert result == ('hello', {'system_prompt': 'Be concise, clear, and helpful'})

    def test_system_prompt_with_overrides(self):
        result = match_prefix('om,t=high,[You are a bot] hello', 'om')
        assert result is not None
        assert result == ('hello', {'t': 'high', 'system_prompt': 'You are a bot'})

    def test_system_prompt_escaped_brackets(self):
        result = match_prefix(r'om,[JSON like \[1,2,3\]] hello', 'om')
        assert result is not None
        assert result == ('hello', {'system_prompt': 'JSON like [1,2,3]'})

    def test_system_prompt_with_template(self):
        result = match_prefix('om,[Time is {current_time}] hello', 'om')
        assert result is not None
        assert result == ('hello', {'system_prompt': 'Time is {current_time}'})

    def test_system_prompt_no_text(self):
        result = match_prefix('om,[You are a bot]', 'om')
        assert result is not None
        assert result == ('', {'system_prompt': 'You are a bot'})

    def test_system_prompt_dollar_not_a_delim(self):
        # $ is not a delimiter, so the entire rest is treated as overrides
        result = match_prefix('om,[You are a bot]$hello', 'om')
        assert result is not None
        # '[You are a bot]$hello' is a single part that doesn't match [..] pattern, becomes bare key
        assert result == ('', {'[You are a bot]$hello': None})

    # --- append system prompt ---

    def test_system_prompt_append_basic(self):
        result = match_prefix('om,+[Reply in JSON] hello', 'om')
        assert result is not None
        assert result == ('hello', {'system_prompt_append': 'Reply in JSON'})

    def test_system_prompt_append_with_overrides(self):
        result = match_prefix('om,t=high,+[Be concise] hello', 'om')
        assert result is not None
        assert result == ('hello', {'t': 'high', 'system_prompt_append': 'Be concise'})

    def test_system_prompt_append_no_text(self):
        result = match_prefix('om,+[Reply in JSON]', 'om')
        assert result is not None
        assert result == ('', {'system_prompt_append': 'Reply in JSON'})

    def test_system_prompt_append_with_commas(self):
        result = match_prefix('om,+[Be concise, clear, helpful] hello', 'om')
        assert result is not None
        assert result == ('hello', {'system_prompt_append': 'Be concise, clear, helpful'})
