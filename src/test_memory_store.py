from __future__ import annotations

from memory_store import estimate_tokens


def test_estimate_tokens_empty_is_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0


def test_estimate_tokens_approximates_chars_over_four() -> None:
    # "abcd" -> 1 token, "abcdefgh" -> 2 tokens (ceil)
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_counts_characters_not_bytes() -> None:
    # "Đà Nẵng" is 7 characters but 9 UTF-8 bytes. Characters must win (2 tokens),
    # otherwise every Vietnamese string is silently over-counted.
    assert len("Đà Nẵng") == 7
    assert len("Đà Nẵng".encode("utf-8")) == 9
    assert estimate_tokens("Đà Nẵng") == 2