"""Public data types returned by toolpair."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

Message = Dict[str, Any]


class IssueCode(str, Enum):
    """The kinds of problem toolpair detects in a message history."""

    MISSING_RESULT = "missing_result"
    """A tool call has no result anywhere in the history."""

    MISPLACED_RESULT = "misplaced_result"
    """A tool call's result exists, but not directly after the call."""

    ORPHAN_RESULT = "orphan_result"
    """A tool result references a call id that does not exist."""

    DUPLICATE_RESULT = "duplicate_result"
    """More than one result answers the same tool call."""

    RESULT_NOT_FIRST = "result_not_first"
    """(Anthropic) Non-tool_result content comes before tool_result blocks."""


@dataclass(frozen=True)
class Issue:
    """One problem found in a message history."""

    code: IssueCode
    message_index: int
    """Index of the message (in the input list) where the problem is anchored."""
    tool_call_id: Optional[str]
    detail: str

    def __str__(self) -> str:
        tid = f" [{self.tool_call_id}]" if self.tool_call_id else ""
        return f"messages[{self.message_index}] {self.code.value}{tid}: {self.detail}"


class Action(str, Enum):
    """What repair() did."""

    SYNTHESIZED_RESULT = "synthesized_result"
    DROPPED_CALL = "dropped_call"
    DROPPED_RESULT = "dropped_result"
    MOVED_RESULT = "moved_result"
    DROPPED_MESSAGE = "dropped_message"
    REORDERED_CONTENT = "reordered_content"


@dataclass(frozen=True)
class Change:
    """One modification made by repair()."""

    action: Action
    message_index: int
    """Index of the affected message in the *input* list."""
    tool_call_id: Optional[str]
    detail: str

    def __str__(self) -> str:
        tid = f" [{self.tool_call_id}]" if self.tool_call_id else ""
        return f"messages[{self.message_index}] {self.action.value}{tid}: {self.detail}"


@dataclass
class RepairResult:
    """Output of repair(): the fixed messages plus a full audit trail."""

    messages: List[Message]
    changes: List[Change] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    """Issues found in the input before repair."""

    @property
    def changed(self) -> bool:
        return bool(self.changes)


class BudgetTooSmallError(ValueError):
    """Raised by trim() when not even the newest unit fits the budget."""
