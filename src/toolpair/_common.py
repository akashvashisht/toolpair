"""Shared helpers: input validation and format detection."""

from __future__ import annotations

from typing import Any, List, Sequence

from ._types import Message

FORMATS = ("openai", "anthropic")
DEFAULT_PLACEHOLDER = (
    "No result was recorded for this tool call (the call was interrupted or "
    "its result was lost). Treat it as failed."
)


def validate_messages(messages: Any) -> List[Message]:
    """Check the basic shape of the input and return it as a list."""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a list of dicts")
    out = list(messages)
    for i, m in enumerate(out):
        if not isinstance(m, dict):
            raise TypeError(
                f"messages[{i}] is {type(m).__name__}, expected dict. "
                "Convert SDK objects first, e.g. msg.model_dump()."
            )
        if "role" not in m:
            raise ValueError(f"messages[{i}] has no 'role' key")
    return out


def _has_anthropic_blocks(m: Message) -> bool:
    content = m.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in content)


def _has_openai_markers(m: Message) -> bool:
    return m.get("role") == "tool" or "tool_calls" in m


def detect_format(messages: Sequence[Message]) -> str:
    """Guess whether a history uses the OpenAI Chat Completions or Anthropic
    Messages tool format.

    Returns "openai" when there are no tool markers at all; both formats treat
    plain text messages the same, so the choice makes no difference then.
    """
    openai = any(_has_openai_markers(m) for m in messages)
    anthropic = any(_has_anthropic_blocks(m) for m in messages)
    if openai and anthropic:
        raise ValueError(
            "History mixes OpenAI tool markers (role 'tool' / 'tool_calls') and "
            "Anthropic tool blocks (tool_use / tool_result). Pass format= "
            "explicitly or convert it to a single format first."
        )
    return "anthropic" if anthropic else "openai"


def resolve_format(messages: Sequence[Message], fmt: str) -> str:
    if fmt == "auto":
        return detect_format(messages)
    if fmt not in FORMATS:
        raise ValueError(f"format must be 'auto', 'openai' or 'anthropic', got {fmt!r}")
    return fmt
