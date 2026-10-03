from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config

# Column headers required by the lab specification.
HEADERS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read a benchmark dataset from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    """Case-fold and strip diacritics so scoring is not defeated by tone marks.

    Vietnamese text is written with combining diacritics, so an exact substring match on
    "Huế" vs "Hue" would silently score 0 for a correct answer.
    """

    decomposed = unicodedata.normalize("NFD", str(text).lower())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.split())


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected facts present in the answer: 0.0, 0.5, or 1.0."""

    if not expected:
        return 1.0
    haystack = _normalize(answer)
    hits = sum(1 for item in expected if _normalize(item) in haystack)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline proxy for answer quality.

    Blends how much of the expected content is covered with concision and structure, so a
    verbose answer cannot win purely by containing the right strings. This is a proxy, not
    a judgement of correctness — with a real LLM available, ``LabConfig.judge_model`` is
    the intended upgrade path.
    """

    if not str(answer).strip():
        return 0.0

    coverage = recall_points(answer, expected)
    words = len(answer.split())
    concision = 1.0 if words <= 60 else max(0.0, 1.0 - (words - 60) / 120.0)
    structured = any(
        marker in answer for marker in ("\n-", "1.", "2.", "3.", "- ")
    )
    structure = 1.0 if structured else 0.6

    return round(0.6 * coverage + 0.2 * concision + 0.2 * structure, 3)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config: LabConfig,
) -> BenchmarkRow:
    """Evaluate one agent over a dataset and return its aggregate metrics.

    Every conversation is replayed in a fresh ``-main`` thread, then its recall questions
    are asked in a separate ``-recall`` thread. That separation is what makes the recall
    column measure long-term memory rather than same-thread recall.
    """

    agent_tokens = 0
    prompt_tokens = 0
    compactions = 0
    memory_bytes = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        main_thread = f"{conversation['id']}-main"
        recall_thread = f"{conversation['id']}-recall"

        for turn in conversation["turns"]:
            result = agent.reply(user_id, main_thread, turn)
            agent_tokens += int(result["agent_tokens"])
            prompt_tokens += int(result["prompt_tokens"])

        for question in conversation.get("recall_questions", []):
            answer = str(agent.reply(user_id, recall_thread, question["question"])["text"])
            expected = list(question.get("expected_contains", []))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        compactions += int(agent.compaction_count(main_thread))
        # Baseline has no persistent memory at all; that contrast is the point.
        if hasattr(agent, "memory_file_size"):
            memory_bytes = max(memory_bytes, int(agent.memory_file_size(user_id)))

    count = len(recall_scores)
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / count, 3) if count else 0.0,
        response_quality=round(sum(quality_scores) / count, 3) if count else 0.0,
        memory_growth_bytes=memory_bytes,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Render comparison rows as a table with the six required metric columns."""

    try:
        from tabulate import tabulate

        table = [[
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.3f}",
            f"{row.response_quality:.3f}",
            row.memory_growth_bytes,
            row.compactions,
        ] for row in rows]
        return tabulate(table, headers=HEADERS, tablefmt="github")
    except ImportError:
        lines = [" | ".join(HEADERS)]
        lines.append("-|-".join("-" * len(h) for h in HEADERS))
        for row in rows:
            lines.append(" | ".join([
                row.agent_name,
                str(row.agent_tokens_only),
                str(row.prompt_tokens_processed),
                f"{row.recall_score:.3f}",
                f"{row.response_quality:.3f}",
                str(row.memory_growth_bytes),
                str(row.compactions),
            ]))
        return "\n".join(lines)


def _run_suite(config: LabConfig, dataset: str, title: str) -> list[BenchmarkRow]:
    """Run both agents over one dataset against an isolated state directory."""

    import shutil
    import tempfile

    conversations = load_conversations(config.data_dir / dataset)
    rows: list[BenchmarkRow] = []

    for name, agent_cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        # Each agent gets a clean state dir so neither inherits the other's profile.
        state_dir = Path(tempfile.mkdtemp(prefix=f"bench-{name.lower()}-"))
        suite_config = LabConfig(
            base_dir=config.base_dir,
            data_dir=config.data_dir,
            state_dir=state_dir,
            compact_threshold_tokens=config.compact_threshold_tokens,
            compact_keep_messages=config.compact_keep_messages,
            model=config.model,
            judge_model=config.judge_model,
        )
        agent = agent_cls(config=suite_config, force_offline=True)
        rows.append(run_agent_benchmark(name, agent, conversations, suite_config))
        shutil.rmtree(state_dir, ignore_errors=True)

    print(f"\n=== {title} ===")
    print(format_rows(rows))
    return rows


def _print_findings(standard: list[BenchmarkRow], stress: list[BenchmarkRow]) -> None:
    """Print the directional takeaways the lab is asking for."""

    base_s, adv_s = standard[0], standard[1]
    base_x, adv_x = stress[0], stress[1]

    print("\n=== NHẬN XÉT NHANH ===")
    print(
        f"- Recall chéo phiên (chuẩn): Baseline {base_s.recall_score:.3f} -> "
        f"Advanced {adv_s.recall_score:.3f}. Lý do: Advanced đọc User.md ở thread mới, "
        f"Baseline chỉ có memory trong thread."
    )
    print(
        f"- Prompt tokens (chuẩn): Baseline {base_s.prompt_tokens_processed} -> "
        f"Advanced {adv_s.prompt_tokens_processed}. Ở hội thoại ngắn Advanced phải kéo thêm "
        f"User.md mỗi lượt nên có thể tốn hơn."
    )
    print(
        f"- Prompt tokens (stress): Baseline {base_x.prompt_tokens_processed} -> "
        f"Advanced {adv_x.prompt_tokens_processed} sau {adv_x.compactions} lần compact. "
        f"Đây là chỗ compact memory thắng rõ ràng."
    )
    print(
        f"- Memory growth: Baseline {base_s.memory_growth_bytes} byte (không có file) -> "
        f"Advanced {adv_s.memory_growth_bytes} byte. File memory là chi phí thật, "
        f"và là nơi dễ lưu sai fact."
    )


def main() -> None:
    """Run the standard and long-context stress suites and print both tables."""

    config = load_config(Path(__file__).resolve().parent.parent)

    standard = _run_suite(
        config, "conversations.json", "BENCHMARK CHUẨN (data/conversations.json)"
    )
    stress = _run_suite(
        config,
        "advanced_long_context.json",
        "BENCHMARK LONG-CONTEXT STRESS (data/advanced_long_context.json)",
    )
    _print_findings(standard, stress)


if __name__ == "__main__":
    main()
