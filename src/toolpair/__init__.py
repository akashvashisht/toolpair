"""toolpair: keep LLM tool calls and tool results paired.

Check, repair and trim chat histories in the OpenAI Chat Completions and
Anthropic Messages formats so a tool call is never separated from its result,
the cause of errors such as "An assistant message with 'tool_calls' must be
followed by tool messages..." and "tool_use ids were found without
tool_result blocks immediately after".
"""

from ._common import DEFAULT_PLACEHOLDER, detect_format
from ._types import (
    Action,
    BudgetTooSmallError,
    Change,
    Issue,
    IssueCode,
    RepairResult,
)
from .api import approx_tokens, check, is_valid, repair, trim

__version__ = "0.1.0"

__all__ = [
    "Action",
    "BudgetTooSmallError",
    "Change",
    "DEFAULT_PLACEHOLDER",
    "Issue",
    "IssueCode",
    "RepairResult",
    "approx_tokens",
    "check",
    "detect_format",
    "is_valid",
    "repair",
    "trim",
    "__version__",
]
