from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path


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

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        raise NotImplementedError

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        raise NotImplementedError

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        raise NotImplementedError

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # TODO: replace one occurrence inside User.md and return whether it changed.
        raise NotImplementedError

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        raise NotImplementedError


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
