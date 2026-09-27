"""Public API: check(), is_valid(), repair(), trim()."""

from __future__ import annotations

import copy
import json
from types import ModuleType
from typing import Callable, Dict, List, Optional, Sequence

from . import _anthropic, _openai
from ._common import DEFAULT_PLACEHOLDER, resolve_format, validate_messages
from ._types import BudgetTooSmallError, Issue, Message, RepairResult

_IMPL: Dict[str, ModuleType] = {"openai": _openai, "anthropic": _anthropic}
_PINNED_ROLES = ("system", "developer")

TokenCounter = Callable[[Message], int]


def check(messages: Sequence[Message], *, format: str = "auto") -> List[Issue]:
    """Return every tool call/result pairing problem in ``messages``.

    An empty list means the provider will accept the history's tool pairing.
    The input is never modified.
    """
    msgs = validate_messages(messages)
    issues: List[Issue] = _IMPL[resolve_format(msgs, format)].check(msgs)
    return issues


def is_valid(messages: Sequence[Message], *, format: str = "auto") -> bool:
    """True if ``check()`` finds no problems."""
    return not check(messages, format=format)


def repair(
    messages: Sequence[Message],
    *,
    format: str = "auto",
    strategy: str = "synthesize",
    placeholder: str = DEFAULT_PLACEHOLDER,
) -> RepairResult:
    """Fix tool call/result pairing so the provider accepts the history.

    * A result that exists but is in the wrong place is **moved** next to its call.
    * A result with no matching call, or a second result for the same call, is **dropped**.
    * A call with no result anywhere is handled by ``strategy``:
      ``"synthesize"`` (default) adds a placeholder result that tells the model the
      call failed; ``"drop"`` removes the call instead.
    * (Anthropic) tool_result blocks are moved ahead of other content.

    Returns a :class:`RepairResult` with the new messages, a list of every change
    made, and the issues found in the input. The input is never modified.
    """
    if strategy not in ("synthesize", "drop"):
        raise ValueError(f"strategy must be 'synthesize' or 'drop', got {strategy!r}")
    msgs = validate_messages(messages)
    fmt = resolve_format(msgs, format)
    impl = _IMPL[fmt]
    issues = impl.check(msgs)
    fixed, changes = impl.repair(msgs, strategy, placeholder)
    return RepairResult(messages=fixed, changes=changes, issues=issues)


def approx_tokens(message: Message) -> int:
    """A rough, provider-agnostic token estimate: about 4 characters per token
    of the JSON-serialised message, plus 4 for message overhead.

    This is an approximation. Pass a real tokenizer to ``trim()`` if you need
    exact limits.
    """
    return len(json.dumps(message, ensure_ascii=False, default=str)) // 4 + 4


def trim(
    messages: Sequence[Message],
    *,
    max_messages: Optional[int] = None,
    max_tokens: Optional[int] = None,
    token_counter: Optional[TokenCounter] = None,
    format: str = "auto",
    keep_system: bool = True,
    start_on_user: bool = False,
    repair_first: bool = True,
    strategy: str = "synthesize",
) -> List[Message]:
    """Keep the most recent messages that fit the budget, without ever
    separating a tool call from its result.

    Messages are grouped into atomic units (a tool call plus its results is
    one unit). Units are kept newest-first until the next one would exceed
    ``max_messages`` or ``max_tokens``; older units are dropped whole.

    Args:
        max_messages: maximum number of messages in the output (system
            messages included).
        max_tokens: maximum total tokens, measured with ``token_counter``
            (defaults to :func:`approx_tokens`).
        keep_system: always keep ``system``/``developer`` messages (OpenAI) and
            count them against the budget.
        start_on_user: drop leading kept units until the history starts with a
            ``user`` message.
        repair_first: run :func:`repair` before trimming so pre-existing damage
            is fixed too. If False, the input must already be valid.

    Raises:
        BudgetTooSmallError: if even the newest unit (plus pinned system
            messages) does not fit.
    """
    if max_messages is None and max_tokens is None:
        raise ValueError("pass max_messages, max_tokens, or both")
    if max_messages is not None and max_messages < 1:
        raise ValueError("max_messages must be >= 1")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")

    msgs = validate_messages(messages)
    fmt = resolve_format(msgs, format)
    impl = _IMPL[fmt]
    if repair_first:
        msgs = impl.repair(msgs, strategy, DEFAULT_PLACEHOLDER)[0]
    elif impl.check(msgs):
        raise ValueError("history has pairing issues; call repair() first or set repair_first=True")

    count = token_counter or approx_tokens
    all_units = impl.units(msgs)
    pinned = [u for u in all_units if keep_system and len(u) == 1 and msgs[u[0]].get("role") in _PINNED_ROLES]
    pinned_ids = {id(u) for u in pinned}
    body = [u for u in all_units if id(u) not in pinned_ids]

    used_msgs = sum(len(u) for u in pinned)
    used_tokens = sum(count(msgs[i]) for u in pinned for i in u) if max_tokens is not None else 0

    kept: List[List[int]] = []
    for unit in reversed(body):
        n = len(unit)
        t = sum(count(msgs[i]) for i in unit) if max_tokens is not None else 0
        if max_messages is not None and used_msgs + n > max_messages:
            break
        if max_tokens is not None and used_tokens + t > max_tokens:
            break
        kept.append(unit)
        used_msgs += n
        used_tokens += t
    kept.reverse()

    if start_on_user:
        while kept and msgs[kept[0][0]].get("role") != "user":
            kept.pop(0)

    if body and not kept:
        raise BudgetTooSmallError(
            "the budget is too small to keep even the newest message unit "
            f"({len(body[-1])} message(s)) alongside {len(pinned)} pinned system message(s)"
            + (" with start_on_user=True" if start_on_user else "")
        )

    keep_idx = sorted(i for u in (*pinned, *kept) for i in u)
    out = [msgs[i] for i in keep_idx]
    # repair() already returned fresh copies; otherwise copy so callers can
    # safely mutate the result without touching their input.
    return out if repair_first else copy.deepcopy(out)
