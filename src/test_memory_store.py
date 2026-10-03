from __future__ import annotations

import json
from pathlib import Path

from memory_store import (
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    merge_fact_values,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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


def test_extract_extracts_declared_facts() -> None:
    facts = extract_profile_updates("Chào bạn, mình tên là DũngCT.")
    assert facts["name"] == "DũngCT"

    facts = extract_profile_updates("Đồ uống yêu thích là cà phê sữa đá.")
    assert facts["drink"] == "cà phê sữa đá"

    # Only the city is captured, never the leading verb.
    facts = extract_profile_updates("Mình ở Huế và đang làm MLOps engineer.")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"


def test_extract_ignores_questions_and_noise() -> None:
    # A question about the user is not a statement of fact.
    assert extract_profile_updates("Bạn có biết DũngCT không?") == {}
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}

    # Explicitly negated / joking facts must not overwrite the real value.
    assert "location" not in extract_profile_updates(
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày, không phải nơi ở hiện tại."
    )
    assert "profession" not in extract_profile_updates(
        "Có lúc mình đùa rằng hay chuyển sang product manager cho đỡ ngồi canh pipeline, "
        "nhưng đó chỉ là câu đùa."
    )


def test_extract_handles_corrections_as_new_facts() -> None:
    # A correction is a fresh declaration: it must be extracted so that
    # last-writer-wins in User.md can replace the stale value.
    facts = extract_profile_updates(
        "À, mình đính chính một chút: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa."
    )
    assert facts["location"] == "Huế", "fact mới phải thắng, Đà Nẵng là dữ liệu cũ"

    facts = extract_profile_updates("Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    assert facts["profession"] == "MLOps engineer"


def test_extract_accumulates_style_markers() -> None:
    facts = extract_profile_updates("Mình muốn bạn trả lời ngắn gọn, rõ ý và có ví dụ thực tế.")
    assert "ngắn gọn" in facts["style"]
    assert "có ví dụ thực chiến" in facts["style"]

    facts = extract_profile_updates("Mình muốn câu trả lời theo dạng 3 bullet ngắn, có ví dụ thực chiến.")
    assert "3 bullet" in facts["style"]


def test_extract_produces_expected_final_profile_on_real_corpus() -> None:
    """End-to-end guard: the corpus every recall question is graded against."""

    def final_profile(dataset: str) -> dict[str, str]:
        profile: dict[str, str] = {}
        for conv in json.loads((DATA_DIR / dataset).read_text(encoding="utf-8")):
            for turn in conv["turns"]:
                for key, value in extract_profile_updates(turn).items():
                    if key in ("style", "interests"):
                        profile[key] = merge_fact_values(profile.get(key, ""), value)
                    else:
                        profile[key] = value
        return profile

    standard = final_profile("conversations.json")
    assert standard["name"] == "DũngCT"
    assert standard["location"] == "Huế", "Huế phải thắng Đà Nẵng"
    assert standard["profession"] == "MLOps engineer", "MLOps phải thắng backend engineer"
    assert standard["drink"] == "cà phê sữa đá"
    assert standard["food"] == "mì Quảng"
    assert standard["pet"] == "corgi"
    assert "ngắn gọn" in standard["style"]

    stress = final_profile("advanced_long_context.json")
    assert stress["name"] == "DũngCT Stress"
    assert stress["location"] == "Đà Nẵng", "correction cuối trong stress phải thắng Huế"
    assert stress["profession"] == "MLOps engineer"
    assert "3 bullet" in stress["style"]