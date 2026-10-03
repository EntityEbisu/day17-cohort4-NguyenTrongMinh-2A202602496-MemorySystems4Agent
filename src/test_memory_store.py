from __future__ import annotations

from pathlib import Path

from memory_store import UserProfileStore, estimate_tokens


def test_estimate_tokens_empty_is_zero() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("   ") == 0


def test_estimate_tokens_approximates_chars_over_four() -> None:
    # "abcd" -> 1 token, "abcdefgh" -> 2 tokens (ceil)
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2


def test_estimate_tokens_counts_characters_not_bytes() -> None:
    # "Đà Nẵng" is 7 characters but 11 UTF-8 bytes. Characters must win (2 tokens),
    # otherwise every Vietnamese string is silently over-counted.
    assert len("Đà Nẵng") == 7
    assert len("Đà Nẵng".encode("utf-8")) > len("Đà Nẵng")
    assert estimate_tokens("Đà Nẵng") == 2


def test_profile_store_path_is_sanitized(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)
    path = store.path_for("dungct_stress")

    assert path.name == "dungct_stress.md"
    # Path traversal must not escape the profiles directory.
    assert store.path_for("../../etc/passwd").parent == tmp_path


def test_profile_store_read_write_edit_roundtrip(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)

    # Unknown user gets a default profile header rather than an error.
    assert "dungct" in store.read_text("dungct")

    store.write_text("dungct", "# Hồ sơ\n\n- name: DũngCT\n")
    assert "- name: DũngCT" in store.read_text("dungct")
    assert store.file_size("dungct") > 0

    assert store.edit_text("dungct", "DũngCT", "Minh") is True
    assert "- name: Minh" in store.read_text("dungct")
    # A missing substring is reported as "no change", not silently written.
    assert store.edit_text("dungct", "khong-ton-tai", "x") is False


def test_profile_store_facts_and_upsert(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path)

    store.write_text("dungct", "# Hồ sơ\n\n- name: DũngCT\n- _updated: 2026-10-03\n")
    facts = store.facts("dungct")

    assert facts == {"name": "DũngCT"}, "metadata keys starting with _ must be excluded"
    assert store.upsert_fact("dungct", "name", "Minh") is True
    assert store.facts("dungct")["name"] == "Minh"
    assert store.upsert_fact("dungct", "location", "Huế") is True
    assert store.facts("dungct")["location"] == "Huế"

    text = store.read_text("dungct")
    assert text.count("- name:") == 1, "upsert phải thay dòng cũ, không thêm dòng trùng"
    assert "- _updated:" in text