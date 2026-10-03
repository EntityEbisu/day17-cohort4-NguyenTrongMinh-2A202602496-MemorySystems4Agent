from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    It remembers what was said in the current thread and nothing else. There is no
    `User.md`, no compact memory, and no cross-thread state — which is exactly what makes
    it a fair comparison point for :class:`AdvancedAgent`.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Answer one turn, returning text plus the two token counters.

        Offline is the default so the lab runs reproducibly with no API key. Live mode is
        used only when a LangChain agent was actually built.
        """

        if self.langchain_agent is not None and not self.force_offline:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Cumulative tokens this agent generated in the given thread."""

        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        """Cumulative prompt context this thread forced the agent to re-read.

        Baseline never compresses, so this grows with every turn — the cost that compact
        memory exists to avoid.
        """

        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        """Baseline has no compact memory."""

        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Deterministic in-thread behaviour with full token accounting."""

        session = self.sessions.setdefault(thread_id, SessionState())

        # This turn's prompt is the whole thread so far plus the new message: baseline
        # re-reads everything, every turn.
        history_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)
        turn_prompt = history_tokens + estimate_tokens(message)
        session.prompt_tokens_processed += turn_prompt

        text = self._offline_reply(session, message)
        session.messages.append({"role": "user", "content": message})
        session.messages.append({"role": "assistant", "content": text})

        agent_tokens = estimate_tokens(text)
        session.token_usage += agent_tokens
        return {
            "text": text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": turn_prompt,
            "mode": "offline",
        }

    def _offline_reply(self, session: SessionState, message: str) -> str:
        """Answer using only this thread's messages.

        The agent quotes what the user said earlier *in the same thread*. In a new thread
        it has nothing to quote, which is precisely the weakness the benchmark exposes —
        not a bug in this method.
        """

        for earlier in reversed(session.messages):
            if earlier["role"] != "user":
                continue
            preview = " ".join(earlier["content"].split())
            if len(preview) > 80:
                preview = preview[:77] + "..."
            return f"Trong thread này bạn đã nói: {preview}"

        return f"Đã ghi nhận: {' '.join(message.split())[:80]}"

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Live LangChain path, reached only when a real model is configured."""

        result = self.langchain_agent.invoke(          # type: ignore[union-attr]
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": f"{user_id}:{thread_id}"}},
        )
        text = str(result["messages"][-1].content)
        agent_tokens = estimate_tokens(text)
        session = self.sessions.setdefault(thread_id, SessionState())
        session.token_usage += agent_tokens
        return {
            "text": text,
            "agent_tokens": agent_tokens,
            "prompt_tokens": agent_tokens,
            "mode": "live",
        }

    def _maybe_build_langchain_agent(self):
        """Build a real LangChain/LangGraph agent when the dependencies exist.

        Returns ``None`` when LangChain is unavailable or offline mode is forced, so
        importing this module never requires LangChain.
        """

        if self.force_offline:
            return None

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        try:
            model = build_chat_model(self.config.model)
        except Exception:
            return None

        return create_agent(model, tools=[], checkpointer=InMemorySaver())
