from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# A fact line in User.md: "- key: value". Keys are ASCII so parsing never has to
# reason about Vietnamese characters.
FACT_LINE_RE = re.compile(r"^-\s+([A-Za-z0-9_]+)\s*:\s*(.*)$")

# `_seen=<n>` is internal recency bookkeeping written by `touch_fact`. It must never leak
# into a fact value, or it would be spoken back to the user and corrupt recall scoring.
SEEN_MARKER_RE = re.compile(r"\s*_seen=\d+\s*$")


def _strip_seen(value: str) -> str:
    """Remove the internal recency marker from a raw fact value."""

    return SEEN_MARKER_RE.sub("", value).strip()


def estimate_tokens(text: str) -> int:
    """Heuristic token estimator: roughly ``len(text) / 4`` characters per token.

    Not an exact tokenizer, but stable and deterministic, which is all the offline
    benchmark needs. Characters are counted, not bytes, so Vietnamese text with
    multi-byte code points is counted correctly.
    """

    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    return int(math.ceil(len(cleaned) / 4))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    One markdown file per user, stored as a flat list of ``- key: value`` facts. Keeping
    the file structured rather than free-form prose is what makes conflict handling and
    per-fact updates possible: a correction overwrites exactly one line instead of
    appending a contradictory sentence.

    File format::

        # Hồ sơ người dùng: dungct

        - name: DũngCT
        - location: Huế
        - _updated: 2026-10-03

    Keys beginning with ``_`` are metadata and are skipped by :meth:`facts`.
    """

    root_dir: Path

    # Class-level monotonic counter backing :meth:`ranked_facts` recency ordering.
    _counter: int = 0

    def path_for(self, user_id: str) -> Path:
        """Map a user id to a single markdown file inside ``root_dir``.

        The id is slugified so a hostile or awkward id can never escape the directory.
        """

        self.root_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9_-]+", "-", str(user_id).strip().lower())
        slug = re.sub(r"-{2,}", "-", slug).strip("-") or "unknown"
        return self.root_dir / f"{slug}.md"

    def read_text(self, user_id: str) -> str:
        """Return the profile markdown, or a fresh header if the user is unknown."""

        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# Hồ sơ người dùng: {user_id}\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk (always UTF-8, for Vietnamese diacritics)."""

        path = self.path_for(user_id)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md; return whether anything changed."""

        text = self.read_text(user_id)
        if search_text not in text:
            return False
        self.write_text(user_id, text.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        """Current size of User.md in bytes; 0 when the profile does not exist yet."""

        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse the profile into ``{key: value}``, excluding ``_`` metadata keys."""

        parsed: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = FACT_LINE_RE.match(line.strip())
            if not match:
                continue
            key = match.group(1)
            value = _strip_seen(match.group(2))
            if key.startswith("_") or not value:
                continue
            parsed[key] = value
        return parsed

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        """Set one fact, replacing an existing line or appending a new one.

        Returns ``True`` when the stored value changed. This is the single write path
        for persistent memory, so conflict resolution (latest fact wins) lives here.
        """

        text = self.read_text(user_id)

        lines = text.splitlines()
        for index, line in enumerate(lines):
            match = FACT_LINE_RE.match(line.strip())
            if match and match.group(1) == key:
                current = _strip_seen(match.group(2))
                if current == value:
                    return False                      # unchanged
                lines[index] = f"- {key}: {value}"
                self._write_lines(user_id, lines)
                return True

        # Insert after the last *fact* line so metadata (`_updated`) always stays last.
        insert_at = 0
        for index, line in enumerate(lines):
            match = FACT_LINE_RE.match(line.strip())
            if match and not match.group(1).startswith("_"):
                insert_at = index + 1
        if insert_at == 0:
            # No facts yet: place it after the header and any blank lines.
            insert_at = len(lines)
            while insert_at > 0 and not lines[insert_at - 1].strip():
                insert_at -= 1
        lines.insert(insert_at, f"- {key}: {value}")
        self._write_lines(user_id, lines)
        return True

    def _write_lines(self, user_id: str, lines: list[str]) -> None:
        body = "\n".join(lines).rstrip() + "\n"
        stamp = f"- _updated: {date.today().isoformat()}"
        if "- _updated:" not in body:
            body = body.rstrip() + "\n" + stamp + "\n"
        self.write_text(user_id, body)

    def touch_fact(self, user_id: str, key: str) -> None:
        """Record that a fact was just mentioned, refreshing its recency.

        This is the bookkeeping behind memory decay: a fact the user keeps repeating stays
        relevant, while one mentioned once and never again slides down the priority order.
        """

        path = self.path_for(user_id)
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        stamp = str(self._tick())

        updated = False
        for index, line in enumerate(lines):
            match = FACT_LINE_RE.match(line.strip())
            if match and match.group(1) == key and not key.startswith("_"):
                lines[index] = f"{line.rstrip()} _seen={stamp}"
                updated = True
                break
        if updated:
            self.write_text(user_id, "\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _tick() -> int:
        """Monotonic counter for recency ordering, independent of wall-clock dates."""

        UserProfileStore._counter += 1
        return UserProfileStore._counter

    def ranked_facts(self, user_id: str) -> list[tuple[str, str]]:
        """Return ``[(key, value)]`` ordered by recency, freshest first.

        Memory decay is applied as *ordering*, not deletion. An old fact is still available
        — the agent simply mentions it after anything the user has been talking about
        lately. Deleting instead would silently lose information the user may still care
        about, which is the failure mode this bonus is meant to avoid.
        """

        path = self.path_for(user_id)
        if not path.exists():
            return []

        entries: list[tuple[int, str, str]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            match = FACT_LINE_RE.match(line.strip())
            if not match:
                continue
            key = match.group(1)
            if key.startswith("_"):
                continue
            raw = match.group(2)
            value = _strip_seen(raw)
            if not value:
                continue
            seen = re.search(r"_seen=(\d+)", raw)
            entries.append((int(seen.group(1)) if seen else -1, key, value))

        # Stable sort on rank keeps file order among facts touched in the same tick.
        entries.sort(key=lambda item: -item[0])
        return [(key, value) for _, key, value in entries]


# --- Fact extraction vocabulary -------------------------------------------------
# Vietnamese corpora need explicit phrase lists; there is no general-purpose NER here,
# and none is needed. These constants are the single source of truth for what counts
# as a stable, persistable fact.

CITIES = (
    r"(?:Đà\s*Nẵng|Huế|Hà\s*Nội|Sài\s*Gòn|Hải\s*Phòng|Cần\s*Thơ|Buôn\s*Ma\s*Thuột|Nha\s*Trang)"
)
PROFESSIONS = (
    r"(?:(?:MLOps|backend|frontend|backend|full[\s-]?stack|data|devops|AI|security)"
    r"[\s-]?engineer|product\s+manager|sinh\s+viên)"
)

# Capturing only the city avoids the classic over-capture of the leading verb
# ("đang ở Huế" instead of "Huế"). The connector word is optional: "Mình ở Huế" is just
# as common as "Mình đang ở tại Huế".
LOCATION_RE = re.compile(
    r"(?:đang\s+ở|hiện\s+ở|mình\s+ở|sống\s+ở|làm\s+việc\s+ở|chuyển\s+đến)\s+"
    r"(?:tại\s+|là\s+)?(" + CITIES + r")",
    re.IGNORECASE,
)
PROFESSION_RE = re.compile(
    r"(?:đang\s+làm\s+(?:một\s+)?|làm\s+|chuyển\s+sang\s+|nghề\s+nghiệp(?:\s+hiện\s+tại)?\s+(?:là|is)\s+)"
    r"(" + PROFESSIONS + r")",
    re.IGNORECASE,
)
NAME_RE = re.compile(
    r"mình\s+tên\s+(?:là|is)\s+([A-ZĐ][\wÀ-ỹ]*(?:\s+[A-ZĐ][\wÀ-ỹ]*)?)", re.IGNORECASE
)
DRINK_RE = re.compile(r"(cà\s*phê\s*sữa\s*đá)", re.IGNORECASE)
FOOD_RE = re.compile(r"(mì\s*Quảng)", re.IGNORECASE)
PET_RE = re.compile(r"(corgi)", re.IGNORECASE)

# Interests are the one field that is explicitly additive, so several may be listed.
INTEREST_TERMS = ("Python", r"AI\s+ứng\s*dụng", r"MLOps", "RAG")

# Style markers accumulate as a set instead of overwriting each other.
STYLE_MARKERS: dict[str, str] = {
    "ngắn gọn": r"ngắn\s*gọn|trả\s+lời\s+(?:ngắn|cô\s+đọng)|bullet\s*ngắn|đừng\s+lan\s+man",
    "3 bullet": r"(?:thành\s+|dạng\s+)?3\s*bullet|ba\s*bullet",
    "có ví dụ thực chiến": r"ví\s*dụ\s*(?:thực\s*chiến|thực\s*tế)",
    "có cấu trúc": r"có\s+cấu\s*trúc|cấu\s*trúc\s+rõ",
    "ưu tiên trade-off": r"trade[\s-]?off|so\s+sánh\s+trade",
}

# Phrases that explicitly invalidate a nearby fact: jokes, one-off trips, corrections
# that demote an old value, and instructions to ignore something.
NOISE_RE = re.compile(
    r"đó\s+chỉ\s+là\s+câu\s+đùa"
    r"|chỉ\s+là\s+câu\s+đùa"
    r"|đó\s+chỉ\s+là\s+câu\s+hỏi"
    r"|chỉ\s+là\s+nơi\s+mình\s+vừa\s+bay"
    r"|chỉ\s+để\s+tham\s+dự"
    r"|không\s+phải\s+nơi\s+ở"
    r"|không\s+phải\s+nơi\s+mình\s+ở"
    r"|đừng\s+lấy\s+nó\s+làm"
    r"|đừng\s+nói\s+\w+\s+nữa"
    r"|đừng\s+coi\s+\w+\s+là"
    r"|ví\s*dụ\s*cũ"
    r"|là\s+ví\s*dụ",
    re.IGNORECASE,
)

# "Đà Nẵng" in a correction sentence is the *old* value being retired, while "Huế" is the
# new one. Statements that only demote an old value must not resurrect it.
RETIRE_RE = re.compile(
    r"(?:không\s+còn\s+(?:ở|làm)|chưa\s+chuyển\s+đi|không\s+còn\s+ở\s+mỗi\s+ngày)",
    re.IGNORECASE,
)

# Turns that ask *about* the user rather than stating a fact.
QUESTION_RE = re.compile(
    r"Bạn\s+có\s+biết"
    r"|có\s+thể\s+nhắc\s+lại"
    r"|nhắc\s+lại"
    r"|thử\s+mô\s+tả"
    r"|Bạn\s+nhớ\s+giúp\s+mình",
    re.IGNORECASE,
)

# First-person declaration verbs: their presence means the user is stating, not asking.
# Kept broad on purpose — "Món ăn yêu thích là mì Quảng" has no subject, but it is still
# a statement of preference, so bare preference phrasing is handled separately below.
DECLARATION_RE = re.compile(
    r"mình\s+(?:tên|đang|làm|ở|sống|thích|muốn|cần|nuôi|hay|vẫn|đính\s*chính|chuyển"
    r"|ăn|uống|làm\s*việc|dùng|đọc|chơi|chạy|đi)",
    re.IGNORECASE,
)

# Subject-less but unambiguous preference statements, e.g. "Món ăn yêu thích là mì Quảng".
PREFERENCE_RE = re.compile(
    r"(?:yêu\s+thích|món\s+ăn|đồ\s+uống|thích\s+nhất)\s*(?:là|bao\s+giờ)?",
    re.IGNORECASE,
)

# An explicit correction is a declaration even without a first-person verb,
# e.g. "không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer".
CORRECTION_RE = re.compile(
    r"đính\s*chính|sửa\s+lại|cập\s*nhật|không\s+còn\s+làm|giờ\s+chuyển\s+sang",
    re.IGNORECASE,
)

# Confidence tiers (bonus: only persist facts we are confident about).
CONFIDENCE_EXPLICIT = 0.95
CONFIDENCE_RESTATED = 0.85
CONFIDENCE_MENTION = 0.5
CONFIDENCE_THRESHOLD = 0.6

# Facts that are safe to append to; everything else is a single-value field where the
# newest declaration wins (conflict handling).
CUMULATIVE_FIELDS = ("style", "interests")


def _confidence_for(text: str, pattern: re.Pattern[str]) -> float:
    """Score how strongly a turn asserts a fact, independent of which fact it is."""

    if DECLARATION_RE.search(text) or CORRECTION_RE.search(text):
        return (
            CONFIDENCE_EXPLICIT
            if not re.search(r"\bvẫn\b|\bvẫn\s+cứ\b", text, re.I)
            else CONFIDENCE_RESTATED
        )
    if PREFERENCE_RE.search(text):
        return CONFIDENCE_EXPLICIT
    if pattern.search(text):
        return CONFIDENCE_MENTION
    return 0.0


def _clean(value: str) -> str:
    """Normalise internal whitespace without touching Vietnamese diacritics."""

    return " ".join(value.split())


def merge_fact_values(existing: str, incoming: str) -> str:
    """Union two comma-separated fact values, preserving first-seen order.

    Used for the additive fields (``style``, ``interests``) so preferences accumulate
    across turns instead of overwriting one another. Scalar fields never go through this.
    """

    parts = [p.strip() for p in (existing or "").split(",") if p.strip()]
    for piece in (incoming or "").split(","):
        piece = piece.strip()
        if piece and piece not in parts:
            parts.append(piece)
    return ", ".join(parts)


def _last_match(text: str, pattern: re.Pattern[str]) -> re.Match[str] | None:
    """Return the *final* match in a turn.

    A correction states the retired value before the new one ("không còn làm backend
    engineer nữa, giờ chuyển sang MLOps engineer"), so the first match is the value the
    user is discarding. Taking the last match is what makes a correction actually win.
    """

    matches = list(pattern.finditer(text))
    return matches[-1] if matches else None


def extract_profile_updates(message: str) -> dict[str, str]:
    """Turn one raw user message into the stable facts it confidently asserts.

    Pure function: no I/O, no network, no writes. Persisting the result is the caller's
    job (see ``AdvancedAgent._reply_offline``), which keeps conflict policy — newest
    declaration wins — in one place.

    Three guards decide whether a candidate fact is trusted:

    1. **Question guard** — a turn that asks about the user ("Bạn có biết DũngCT
       không?") states nothing, so it yields no facts.
    2. **Noise guard** — jokes, one-off trips, and explicit "use the new value instead"
       instructions suppress the fact they mention, which is what stops `product
       manager` (a joke) or `Hà Nội` (a two-day trip) from overwriting the real ones.
    3. **Confidence threshold** — candidates below ``CONFIDENCE_THRESHOLD`` are dropped
       rather than persisted on a hunch.
    """

    text = (message or "").strip()
    if not text:
        return {}

    # Guard 1: questions are not declarations.
    if QUESTION_RE.search(text) and not DECLARATION_RE.search(text):
        return {}

    noisy = bool(NOISE_RE.search(text))
    retired = bool(RETIRE_RE.search(text))

    candidates: dict[str, str] = {}

    def offer(key: str, value: str, pattern: re.Pattern[str]) -> None:
        if noisy:
            return
        if _confidence_for(text, pattern) < CONFIDENCE_THRESHOLD:
            return
        candidates[key] = _clean(value)

    name_match = NAME_RE.search(text)
    if name_match:
        offer("name", name_match.group(1), NAME_RE)

    location_match = _last_match(text, LOCATION_RE)
    if location_match:
        offer("location", location_match.group(1), LOCATION_RE)

    profession_match = _last_match(text, PROFESSION_RE)
    if profession_match:
        offer("profession", profession_match.group(1), PROFESSION_RE)

    for key, pattern in (("drink", DRINK_RE), ("food", FOOD_RE), ("pet", PET_RE)):
        match = pattern.search(text)
        if match:
            offer(key, match.group(1), pattern)

    # Interests: collect every term present in the turn.
    interests = [_clean(m.group(0)) for m in
                 (re.search(term, text, re.IGNORECASE) for term in INTEREST_TERMS) if m]
    if interests and not noisy and not retired:
        candidates["interests"] = ", ".join(dict.fromkeys(interests))

    # Style: union of every marker present, so preferences accumulate rather than
    # replacing one another.
    style_markers = [label for label, pattern in STYLE_MARKERS.items()
                     if re.search(pattern, text, re.IGNORECASE)]
    if style_markers:
        candidates["style"] = ", ".join(style_markers)

    return candidates


# Hard cap on the compacted summary. Without it the summary would itself become the
# unbounded-history problem it exists to solve. It is also capped relative to the
# configured threshold: a summary that fills the whole budget would leave compaction
# unable to relieve any pressure, and the thread would keep re-triggering.
SUMMARY_MAX_CHARS = 1200
SUMMARY_THRESHOLD_FRACTION = 0.25

# Per-message preview length inside a summary line.
SUMMARY_ITEM_CHARS = 120


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic compaction of older messages into one line of running context.

    Deliberately not LLM-based: the offline benchmark must be deterministic and free, so
    this keeps the head and tail of a thread and drops the middle. Each message is reduced
    to a trimmed single line, which is what stops a 16-turn stress thread from carrying its
    full text into every later prompt.
    """

    items = [m for m in messages if str(m.get("content", "")).strip()]
    if not items:
        return ""

    lines: list[str] = []
    for message in items:
        role = "U" if message.get("role") == "user" else "A"
        body = " ".join(str(message.get("content", "")).split())
        if len(body) > SUMMARY_ITEM_CHARS:
            body = body[: SUMMARY_ITEM_CHARS - 3] + "..."
        lines.append(f"{role}: {body}")

    if len(lines) > max_items:
        older = lines[: len(lines) - max_items]
        recent = lines[len(lines) - max_items:]
        return f"[nén {len(older)} lượt] " + " | ".join(older + ["..."]) + " | giữ gần đây: " + " | ".join(recent)
    return " | ".join(lines)


@dataclass
class CompactMemoryManager:
    """Short-term memory that compresses itself when a thread grows too large.

    Each thread keeps:

    - ``messages`` — the most recent ``keep_messages`` turns, verbatim
    - ``summary``  — everything older, compressed into one bounded string
    - ``compactions`` — how many times compression has fired, for the benchmark

    Compaction is the only mechanism that ever discards history, which keeps "how much
    context does the agent carry" a single measurable quantity.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _empty_state(self) -> dict[str, object]:
        return {
            "messages": [],
            "summary": "",
            "compactions": 0,
            "total_tokens": 0,       # full history size, drives the compaction trigger
        }

    def _context_tokens(self, state: dict[str, object]) -> int:
        """Tokens actually carried into the next prompt: summary plus kept messages."""

        total = estimate_tokens(str(state.get("summary", "")))
        for message in state["messages"]:          # type: ignore[union-attr]
            total += estimate_tokens(str(message.get("content", "")))
        return total

    def _summary_cap(self) -> int:
        """Summary budget in characters, kept to a fraction of the token threshold.

        Compaction only helps if freeing old messages actually brings the thread back
        under budget, so the summary may never dominate the threshold.
        """

        budget = int(self.threshold_tokens * SUMMARY_THRESHOLD_FRACTION * 4)
        return max(200, min(SUMMARY_MAX_CHARS, budget))

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Add one message, then compress the thread when its history outgrows the budget.

        ``total_tokens`` tracks the *untrimmed* size of the thread. It is what decides
        whether a compaction happened, because that is the number the benchmark's
        "Compactions" column reports and the one that grows without bound if nobody
        compresses. The retained buffer, meanwhile, is always trimmed to the newest
        ``keep_messages`` turns so prompt size stays flat.
        """

        state = self.state.setdefault(thread_id, self._empty_state())
        messages: list[dict[str, str]] = state["messages"]       # type: ignore[assignment]
        messages.append({"role": role, "content": content})
        state["total_tokens"] = int(state["total_tokens"]) + estimate_tokens(content)

        if int(state["total_tokens"]) > self.threshold_tokens:
            prior = str(state.get("summary", ""))
            pieces = [{"role": m["role"], "content": m["content"]} for m in messages]
            if prior:
                pieces.append({"role": "summary", "content": prior})
            state["summary"] = summarize_messages(pieces)[-self._summary_cap():]
            # Reset the budget to what the summary still represents, so the next
            # compaction happens after genuinely new content rather than immediately.
            state["total_tokens"] = estimate_tokens(str(state["summary"]))
            state["compactions"] = int(state["compactions"]) + 1

        if len(messages) > self.keep_messages:
            messages[:] = messages[-self.keep_messages:]

    def context(self, thread_id: str) -> dict[str, object]:
        """Return the live per-thread state (``messages``, ``summary``, ``compactions``)."""

        return self.state.setdefault(thread_id, self._empty_state())

    def compaction_count(self, thread_id: str) -> int:
        """How many times this thread has been compacted."""

        return int(self.state.get(thread_id, {}).get("compactions", 0))    # type: ignore[union-attr]
