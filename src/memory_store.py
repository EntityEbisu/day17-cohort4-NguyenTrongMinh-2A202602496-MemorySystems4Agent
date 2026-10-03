from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# A fact line in User.md: "- key: value". Keys are ASCII so parsing never has to
# reason about Vietnamese characters.
FACT_LINE_RE = re.compile(r"^-\s+([A-Za-z0-9_]+)\s*:\s*(.*)$")


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
            key, value = match.group(1), match.group(2).strip()
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
        new_line = f"- {key}: {value}"

        lines = text.splitlines()
        for index, line in enumerate(lines):
            match = FACT_LINE_RE.match(line.strip())
            if match and match.group(1) == key:
                if line.strip() == new_line:
                    return False
                lines[index] = new_line
                self._write_lines(user_id, lines)
                return True

        # Append after the last fact line so the file keeps header, facts, metadata.
        insert_at = len(lines)
        for index in range(len(lines) - 1, -1, -1):
            if FACT_LINE_RE.match(lines[index].strip()):
                insert_at = index + 1
                break
        lines.insert(insert_at, new_line)
        self._write_lines(user_id, lines)
        return True

    def _write_lines(self, user_id: str, lines: list[str]) -> None:
        body = "\n".join(lines).rstrip() + "\n"
        stamp = f"- _updated: {date.today().isoformat()}"
        if "- _updated:" not in body:
            body = body.rstrip() + "\n" + stamp + "\n"
        self.write_text(user_id, body)


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    raise NotImplementedError


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    raise NotImplementedError


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        raise NotImplementedError

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        raise NotImplementedError

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        raise NotImplementedError
