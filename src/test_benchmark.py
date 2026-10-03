from __future__ import annotations

import json
from pathlib import Path

from benchmark import (
    format_rows,
    heuristic_quality,
    load_conversations,
    recall_points,
    run_agent_benchmark,
)
from config import load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_load_conversations_reads_json(tmp_path: Path) -> None:
    convs = load_conversations(DATA_DIR / "conversations.json")

    assert len(convs) == 10
    assert convs[0]["user_id"] == "dungct"
    assert "recall_questions" in convs[0]

    stress = load_conversations(DATA_DIR / "advanced_long_context.json")
    assert len(stress) == 1
    assert stress[0]["user_id"] == "dungct_stress"


def test_recall_points_scores_coverage() -> None:
    assert recall_points("Bạn là DũngCT", ["DũngCT"]) == 1.0
    assert recall_points("Bạn là DũngCT", ["DũngCT", "mì Quảng"]) == 0.5
    assert recall_points("Không có gì", ["DũngCT"]) == 0.0
    assert recall_points("bất kỳ", []) == 1.0


def test_recall_points_ignores_diacritics_and_case() -> None:
    """Diacritic-blind matching: a tone-mark difference must not zero a correct answer."""

    assert recall_points("Bạn đang ở Hue", ["Huế"]) == 1.0
    assert recall_points("đồ uống: cà phê sữa đá", ["Cà Phê Sữa Đá"]) == 1.0


def test_heuristic_quality_penalises_verbosity() -> None:
    expected = ["DũngCT"]
    short = heuristic_quality("Tên của bạn là DũngCT.", expected)
    long = heuristic_quality("DũngCT " * 200, expected)

    assert short > long, "trả lời dài dòng không được thắng trả lời súc tích"
    assert heuristic_quality("", expected) == 0.0


def test_run_agent_benchmark_advances_recall_over_baseline(tmp_path: Path) -> None:
    """The lab's headline result: persistent memory must win cross-session recall."""

    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent
    from test_agents import make_config

    config = make_config(tmp_path)
    convs = load_conversations(DATA_DIR / "conversations.json")

    baseline_row = run_agent_benchmark(
        "Baseline", BaselineAgent(config=config, force_offline=True), convs, config
    )
    advanced_row = run_agent_benchmark(
        "Advanced", AdvancedAgent(config=config, force_offline=True), convs, config
    )

    assert advanced_row.recall_score > baseline_row.recall_score, \
        "Advanced phải recall tốt hơn Baseline"
    assert baseline_row.memory_growth_bytes == 0, "baseline không có file memory"
    assert advanced_row.memory_growth_bytes > 0
    assert baseline_row.compactions == 0
    assert advanced_row.agent_tokens_only > 0
    assert advanced_row.prompt_tokens_processed > 0


def test_run_agent_benchmark_reports_compactions_on_stress(tmp_path: Path) -> None:
    from agent_advanced import AdvancedAgent
    from test_agents import make_config

    config = make_config(tmp_path)
    stress = load_conversations(DATA_DIR / "advanced_long_context.json")

    row = run_agent_benchmark(
        "Advanced", AdvancedAgent(config=config, force_offline=True), stress, config
    )

    assert row.compactions > 0, "stress benchmark phải kích hoạt compact memory"
    assert 0.0 < row.recall_score <= 1.0


def test_intent_routing_covers_question_phrasings() -> None:
    """Every phrasing used by the benchmark datasets must route to the right fact."""

    from agent_advanced import INTENT_PATTERNS

    cases = [
        ("Hiện tại mình làm nghề gì và mình còn ở Huế không?", {"profession", "location"}),
        ("Hiện tại mình đang ở đâu?", {"location"}),
        ("Mình thích style trả lời như thế nào và hiện đang ở đâu?", {"location", "style"}),
        ("Bạn biết DũngCT là ai không? Hãy nhắc tên và mối quan tâm chính của mình.",
         {"name", "interests"}),
        ("Đồ uống và món ăn yêu thích của mình là gì?", {"drink", "food"}),
    ]

    for question, expected in cases:
        hits = {key for key, pattern in INTENT_PATTERNS.items() if pattern.search(question)}
        assert expected <= hits, f"thiếu intent {expected - hits} cho câu: {question}"


def test_recall_is_measured_after_the_whole_dataset(tmp_path: Path) -> None:
    """Recall must reflect the full profile, not just what one conversation supplied.

    Asking conv-01's questions before conv-03/conv-06 have been replayed would score the
    agent against a half-built profile and understate persistent memory.
    """

    from agent_advanced import AdvancedAgent
    from test_agents import make_config

    config = make_config(tmp_path)
    convs = load_conversations(DATA_DIR / "conversations.json")

    row = run_agent_benchmark(
        "Advanced", AdvancedAgent(config=config, force_offline=True), convs, config
    )

    assert row.recall_score >= 0.9, (
        "sau khi nghe toàn bộ 10 hội thoại, agent phải nhớ được gần như mọi fact; "
        f"recall={row.recall_score}"
    )


def test_format_rows_prints_all_six_columns() -> None:
    from benchmark import BenchmarkRow

    rows = [
        BenchmarkRow("Baseline", 100, 2000, 0.0, 0.3, 0, 0),
        BenchmarkRow("Advanced", 150, 900, 1.0, 0.9, 412, 6),
    ]
    table = format_rows(rows)

    for header in ["Agent tokens only", "Prompt tokens processed", "Cross-session recall",
                   "Response quality", "Memory growth (bytes)", "Compactions"]:
        assert header in table, f"thiếu cột bắt buộc: {header}"
    assert "Baseline" in table and "Advanced" in table