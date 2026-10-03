from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def make_config(tmp_path: Path):
    """Isolated config for tests: state inside tmp_path, low compact threshold."""

    base = load_config(base_dir=tmp_path)
    return replace(
        base,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=400,
        compact_keep_messages=4,
    )


# --- BaselineAgent -------------------------------------------------------------


def test_baseline_remembers_within_a_thread(tmp_path: Path) -> None:
    agent = BaselineAgent(config=make_config(tmp_path), force_offline=True)

    agent.reply("dungct", "t1", "Mình tên là DũngCT.")

    reply = agent.reply("dungct", "t1", "Bạn nhớ tên mình không?")
    assert "DũngCT" in reply["text"], "trong cùng thread, baseline phải nhớ được"


def test_baseline_forgets_across_threads(tmp_path: Path) -> None:
    """The rubric's core fairness check: baseline must NOT recall in a new thread."""

    agent = BaselineAgent(config=make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Mình tên là DũngCT.")

    reply = agent.reply("dungct", "t2", "Mình tên gì?")

    assert "DũngCT" not in reply["text"], "baseline không được nhớ fact qua thread mới"
    assert agent.compaction_count("t1") == 0, "baseline không có compact memory"


def test_baseline_accounting_is_per_thread(tmp_path: Path) -> None:
    agent = BaselineAgent(config=make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Mình tên là DũngCT.")
    agent.reply("dungct", "t1", "Mình ở Huế.")

    assert agent.token_usage("t1") > 0
    assert agent.prompt_token_usage("t1") > 0
    assert agent.token_usage("chưa-có") == 0
    assert agent.prompt_token_usage("chưa-có") == 0


def test_baseline_prompt_load_grows_with_history(tmp_path: Path) -> None:
    """Baseline has no compression, so prompt cost must keep growing."""

    agent = BaselineAgent(config=make_config(tmp_path), force_offline=True)
    for i in range(8):
        agent.reply("dungct", "t1", f"Lượt {i}: " + "nội dung dài " * 20)

    assert agent.prompt_token_usage("t1") > 800


def test_baseline_never_writes_user_md(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = BaselineAgent(config=config, force_offline=True)

    agent.reply("dungct", "t1", "Mình tên là DũngCT.")

    profiles = config.state_dir / "profiles"
    assert not profiles.exists() or not list(profiles.glob("*.md")), \
        "baseline không được tạo persistent memory"


# --- AdvancedAgent -------------------------------------------------------------


def test_advanced_remembers_across_threads(tmp_path: Path) -> None:
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Mình tên là DũngCT.")
    agent.reply("dungct", "t1", "Mình ở Huế và đang làm MLOps engineer.")

    reply = agent.reply("dungct", "t2", "Tên mình là gì, mình ở đâu?")

    assert "DũngCT" in reply["text"]
    assert "Huế" in reply["text"]


def test_advanced_resolves_corrections_to_the_newest_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    agent.reply("dungct", "t1", "Mình ở Đà Nẵng và đang làm backend engineer.")
    agent.reply("dungct", "t1", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")
    agent.reply("dungct", "t1", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")

    reply = agent.reply("dungct", "t2", "Hiện tại mình ở đâu và làm nghề gì?")

    assert "Huế" in reply["text"]
    assert "MLOps engineer" in reply["text"]
    assert "Đà Nẵng" not in reply["text"], "fact cũ đã bị thay thế không được xuất hiện"
    assert "backend engineer" not in reply["text"], "nghề cũ đã bị thay thế không được xuất hiện"


def test_advanced_writes_and_reuses_user_md(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    agent.reply("dungct", "t1", "Mình tên là DũngCT.")

    assert agent.memory_file_size("dungct") > 0
    stored = (config.state_dir / "profiles" / "dungct.md").read_text(encoding="utf-8")
    assert "DũngCT" in stored


def test_advanced_answers_composite_recall_questions(tmp_path: Path) -> None:
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    for turn in [
        "Mình tên là DũngCT.",
        "Mình đang ở Huế.",
        "Đồ uống yêu thích là cà phê sữa đá.",
        "Món ăn yêu thích là mì Quảng.",
        "Mình muốn bạn trả lời ngắn gọn.",
    ]:
        agent.reply("dungct", "t1", turn)

    reply = agent.reply(
        "dungct", "t2",
        "Nhắc lại giúp mình: tên, nơi ở, đồ uống, món ăn và style trả lời.",
    )["text"]

    for expected in ["DũngCT", "Huế", "cà phê sữa đá", "mì Quảng", "ngắn gọn"]:
        assert expected in reply, f"câu hỏi tổng hợp phải trả lời được: {expected}"


def test_advanced_admits_when_it_knows_nothing(tmp_path: Path) -> None:
    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    reply = agent.reply("nguoi-moi", "t1", "Mình tên gì?")
    assert "chưa có thông tin" in reply["text"].lower()


# --- Lab-required behaviours ---------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """User.md must be created, updated, and edited through the agent."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    agent.reply("dungct", "t1", "Mình tên là DũngCT.")
    profile = config.state_dir / "profiles" / "dungct.md"
    assert profile.exists()
    first_size = agent.memory_file_size("dungct")
    assert first_size > 0

    agent.reply("dungct", "t1", "Mình đang ở Huế.")
    assert agent.memory_file_size("dungct") > first_size, "thêm fact phải làm file phình ra"
    stored = profile.read_text(encoding="utf-8")
    assert "DũngCT" in stored and "Huế" in stored


def test_compact_trigger(tmp_path: Path) -> None:
    """Long threads must actually trigger compaction."""

    agent = AdvancedAgent(config=make_config(tmp_path), force_offline=True)
    stress = json.loads((DATA_DIR / "advanced_long_context.json").read_text(encoding="utf-8"))[0]

    for turn in stress["turns"]:
        agent.reply(stress["user_id"], "long", turn)

    assert agent.compaction_count("long") > 0, "compact memory phải thực sự kích hoạt"


def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced remembers across sessions; baseline does not."""

    config = make_config(tmp_path)
    advanced = AdvancedAgent(config=config, force_offline=True)
    baseline = BaselineAgent(config=config, force_offline=True)

    for agent in (advanced, baseline):
        agent.reply("dungct", "session-1", "Mình tên là DũngCT.")

    assert "DũngCT" in advanced.reply("dungct", "session-2", "Mình tên gì?")["text"]
    assert "DũngCT" not in baseline.reply("dungct", "session-2", "Mình tên gì?")["text"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """The lab's headline claim: compaction keeps long-thread prompt cost down."""

    config = make_config(tmp_path)
    stress = json.loads((DATA_DIR / "advanced_long_context.json").read_text(encoding="utf-8"))[0]

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    for turn in stress["turns"]:
        baseline.reply(stress["user_id"], "long", turn)
        advanced.reply(stress["user_id"], "long", turn)

    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long"), \
        "advanced phải xử lý ngữ cảnh dài rẻ hơn baseline"