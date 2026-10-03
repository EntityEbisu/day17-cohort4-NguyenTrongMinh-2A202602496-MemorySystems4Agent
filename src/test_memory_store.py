from __future__ import annotations

from memory_store import estimate_tokens


def test_estimate_tokens_empty_is_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0


def test_estimate_tokens_approximates_chars_over_four() -> None:
    # "abcd" -> 1 token, "abcdefgh" -> 2 tokens (ceil)
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_handles_vietnamese() -> None:
    # 8 chars of multi-byte Vietnamese text must count as 2 tokens, not 8
    assert estimate_tokens("tiếngViệt") == 2