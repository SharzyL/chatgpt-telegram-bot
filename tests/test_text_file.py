"""Deciding whether an attached file is text the model can be given."""

from chatgpt_telegram_bot.utils import decode_text_file


def test_ascii_is_text():
    assert decode_text_file(b'hello\nworld\n') == 'hello\nworld\n'


def test_multibyte_utf8_is_text():
    assert decode_text_file('héllo — 世界'.encode()) == 'héllo — 世界'


def test_an_empty_file_is_empty_text_not_a_rejection():
    assert decode_text_file(b'') == ''


def test_invalid_utf8_is_rejected():
    assert decode_text_file(b'\xff\xfe\x00h') is None


def test_a_nul_byte_is_rejected():
    # decodes cleanly, so the check has to be explicit; it used to be an assert, which
    # escaped both the UnicodeDecodeError handler and the message handler itself
    assert decode_text_file(b'text\x00more') is None


def test_a_file_of_only_nul_bytes_is_rejected():
    assert decode_text_file(b'\x00' * 16) is None
