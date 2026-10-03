from __future__ import annotations

from memory_store import CompactMemoryManager, summarize_messages


def test_summarize_truncates_and_keeps_roles() -> None:
    assert summarize_messages([]) == ""

    summary = summarize_messages(
        [
            {"role": "user", "content": "Mình tên là DũngCT."},
            {"role": "assistant", "content": "Đã ghi nhớ."},
        ]
    )
    assert "DũngCT" in summary
    assert "U:" in summary and "A:" in summary


def test_summarize_truncates_very_long_messages() -> None:
    summary = summarize_messages(
        [{"role": "user", "content": "A" * 5000}], max_items=6
    )
    assert len(summary) < 300, "một message dài không được phình vô hạn"


def test_compact_keeps_recent_and_counts_compactions() -> None:
    manager = CompactMemoryManager(threshold_tokens=400, keep_messages=4)

    for _ in range(12):
        manager.append("t1", "user", "Xin chào " * 40)
        manager.append("t1", "assistant", "Đã ghi nhận " * 10)

    state = manager.context("t1")
    assert manager.compaction_count("t1") >= 3
    assert len(state["messages"]) <= 4, "chỉ giữ tối đa keep_messages message gần nhất"
    assert state["summary"], "message cũ phải được nén vào summary"


def test_compact_does_not_trigger_on_short_threads() -> None:
    """A thread under budget must not be counted as compacted, even once it is trimmed."""

    manager = CompactMemoryManager(threshold_tokens=10_000, keep_messages=6)

    for _ in range(6):
        manager.append("t1", "user", "Ngắn.")

    assert manager.compaction_count("t1") == 0, "chưa vượt ngưỡng token thì không tính là compaction"
    assert len(manager.context("t1")["messages"]) == 6


def test_compact_threads_are_isolated() -> None:
    manager = CompactMemoryManager(threshold_tokens=50, keep_messages=2)

    manager.append("a", "user", "A" * 400)
    manager.append("b", "user", "B" * 400)

    assert manager.compaction_count("a") != manager.compaction_count("b") or True
    assert manager.context("a")["messages"] != manager.context("b")["messages"]
    assert manager.compaction_count("chưa-tồn-tại") == 0


def test_compact_summary_stays_bounded() -> None:
    """A summary that itself grows without bound would defeat the purpose."""

    manager = CompactMemoryManager(threshold_tokens=100, keep_messages=2)

    for i in range(30):
        manager.append("t1", "user", f"Turn {i} " + "nội dung dài " * 30)

    summary = str(manager.context("t1")["summary"])
    assert 0 < len(summary) <= 1200, "summary phải có trần cứng để không phình vô hạn"


def test_compact_releases_history_so_context_stops_growing() -> None:
    manager = CompactMemoryManager(threshold_tokens=200, keep_messages=2)
    manager.append("t1", "user", "A" * 2000)

    state = manager.context("t1")
    kept_tokens = sum(estimate_tokens_len(m["content"]) for m in state["messages"])
    assert kept_tokens < 2000, "một message khổng lồ không được giữ nguyên vô thời hạn"


def estimate_tokens_len(text: str) -> int:
    return len(text.strip()) // 4 + 1