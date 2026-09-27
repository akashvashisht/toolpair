"""Anthropic Messages format.

Rules enforced (quoted from Anthropic's tool-use documentation):
  * "Tool result blocks must immediately follow their corresponding tool use
    blocks in the message history. You cannot include any messages between the
    assistant's tool use message and the user's tool result message."
  * "In the user message containing tool results, the tool_result blocks must
    come FIRST in the content array. Any text must come AFTER all tool results."
  * API errors when broken: "tool_use ids were found without tool_result blocks
    immediately after" and "unexpected `tool_use_id` found in `tool_result`
    blocks ... Each `tool_result` block must have a corresponding `tool_use`
    block in the previous message."

Only client tools (``tool_use`` / ``tool_result``) are paired. Server-tool
blocks (``server_tool_use``, ``web_search_tool_result``, ...) live inside the
assistant message and are left untouched.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from ._types import Action, Change, Issue, IssueCode, Message

Loc = Tuple[int, int]  # (message index, block index)


def _blocks(m: Message) -> List[Any]:
    c = m.get("content")
    return c if isinstance(c, list) else []


def _is_result(b: Any) -> bool:
    return isinstance(b, dict) and b.get("type") == "tool_result"


def use_ids(m: Message) -> List[Optional[str]]:
    """tool_use ids in an assistant message, in order, de-duplicated."""
    if m.get("role") != "assistant":
        return []
    out: List[Optional[str]] = []
    for b in _blocks(m):
        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") not in out:
            out.append(b.get("id"))
    return out


def _next_user(messages: List[Message], i: int) -> Optional[int]:
    j = i + 1
    if j < len(messages) and messages[j].get("role") == "user":
        return j
    return None


def _result_locs(messages: List[Message]) -> Dict[Optional[str], List[Loc]]:
    locs: Dict[Optional[str], List[Loc]] = {}
    for mi, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        for bi, b in enumerate(_blocks(m)):
            if _is_result(b):
                locs.setdefault(b.get("tool_use_id"), []).append((mi, bi))
    return locs


def check(messages: List[Message]) -> List[Issue]:
    issues: List[Issue] = []
    locs = _result_locs(messages)
    all_use: Set[Optional[str]] = set()
    claimed: Set[Loc] = set()
    answered: Set[Optional[str]] = set()

    # Pass 1: results that sit where they belong.
    calls: Dict[int, List[Optional[str]]] = {}
    for i, m in enumerate(messages):
        ids = use_ids(m)
        if not ids:
            continue
        calls[i] = ids
        all_use.update(ids)
        j = _next_user(messages, i)
        if j is None:
            continue
        seen: Set[Optional[str]] = set()
        for bi, b in enumerate(_blocks(messages[j])):
            if _is_result(b) and b.get("tool_use_id") in ids and b.get("tool_use_id") not in seen:
                seen.add(b.get("tool_use_id"))
                claimed.add((j, bi))
                answered.add(b.get("tool_use_id"))

    # Pass 2: calls whose result is missing or elsewhere.
    for i, ids in calls.items():
        j = _next_user(messages, i)
        in_place = (
            {b.get("tool_use_id") for bi, b in enumerate(_blocks(messages[j])) if (j, bi) in claimed}
            if j is not None
            else set()
        )
        for tid in ids:
            if tid in in_place:
                continue
            elsewhere = [loc for loc in locs.get(tid, []) if loc not in claimed]
            if elsewhere:
                issues.append(
                    Issue(
                        IssueCode.MISPLACED_RESULT,
                        i,
                        tid,
                        f"tool_result is in messages[{elsewhere[0][0]}], not in the "
                        "user message directly after the tool_use",
                    )
                )
            else:
                issues.append(
                    Issue(
                        IssueCode.MISSING_RESULT,
                        i,
                        tid,
                        "tool_use has no tool_result anywhere",
                    )
                )

    # Pass 3: stray results and ordering inside user messages.
    misplaced_seen: Set[Optional[str]] = set()
    for mi, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        non_result_seen = False
        reported_order = False
        for bi, b in enumerate(_blocks(m)):
            if not _is_result(b):
                non_result_seen = True
                continue
            if non_result_seen and not reported_order:
                issues.append(
                    Issue(
                        IssueCode.RESULT_NOT_FIRST,
                        mi,
                        b.get("tool_use_id"),
                        "tool_result blocks must come before any other content",
                    )
                )
                reported_order = True
            if (mi, bi) in claimed:
                continue
            tid = b.get("tool_use_id")
            if tid in answered or tid in misplaced_seen:
                issues.append(
                    Issue(
                        IssueCode.DUPLICATE_RESULT,
                        mi,
                        tid,
                        "another tool_result already answers this tool_use",
                    )
                )
            elif tid in all_use:
                misplaced_seen.add(tid)  # reported at the call site
            else:
                issues.append(
                    Issue(
                        IssueCode.ORPHAN_RESULT,
                        mi,
                        tid,
                        "tool_result references a tool_use id that does not exist",
                    )
                )
    return sorted(issues, key=lambda x: x.message_index)


def repair(messages: List[Message], strategy: str, placeholder: str) -> Tuple[List[Message], List[Change]]:
    locs = _result_locs(messages)

    # Plan: pick exactly one tool_result block for every (assistant, id).
    used: Set[Loc] = set()
    assigned: Dict[Tuple[int, Optional[str]], Loc] = {}
    for i, m in enumerate(messages):
        ids = use_ids(m)
        if not ids:
            continue
        j = _next_user(messages, i)
        for tid in ids:
            cands = locs.get(tid, [])
            pick: Optional[Loc] = next((c for c in cands if c[0] == j and c not in used), None)
            if pick is None:
                pick = next((c for c in cands if c[0] > i and c not in used), None)
            if pick is None:
                pick = next((c for c in reversed(cands) if c[0] < i and c not in used), None)
            if pick is not None:
                used.add(pick)
                assigned[(i, tid)] = pick
    answered_ids = {tid for (_, tid) in assigned}

    out: List[Message] = []
    changes: List[Change] = []
    merged_users: Set[int] = set()

    def leftover(mi: int) -> Tuple[List[Any], bool]:
        """Non-result blocks of user message mi, dropping stray results.
        Returns (blocks, had_results)."""
        keep: List[Any] = []
        had = False
        for bi, b in enumerate(_blocks(messages[mi])):
            if not _is_result(b):
                keep.append(copy.deepcopy(b))
                continue
            had = True
            if (mi, bi) in used:
                continue  # emitted next to its tool_use
            tid = b.get("tool_use_id")
            why = (
                "duplicate result for an already-answered tool_use"
                if tid in answered_ids
                else "no matching tool_use"
            )
            changes.append(Change(Action.DROPPED_RESULT, mi, tid, why))
        return keep, had

    for i, m in enumerate(messages):
        if i in merged_users:
            continue
        role = m.get("role")
        ids = use_ids(m)

        if ids:
            new_blocks: List[Any] = []
            results: List[Any] = []
            done: Set[Optional[str]] = set()
            for b in _blocks(m):
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    new_blocks.append(copy.deepcopy(b))
                    continue
                tid = b.get("id")
                loc = assigned.get((i, tid))
                if loc is None and strategy == "drop":
                    changes.append(
                        Change(Action.DROPPED_CALL, i, tid, "removed a tool_use that had no result")
                    )
                    continue
                new_blocks.append(copy.deepcopy(b))
                if tid in done:
                    continue
                done.add(tid)
                if loc is not None:
                    results.append(copy.deepcopy(messages[loc[0]]["content"][loc[1]]))
                    if loc[0] != i + 1:
                        changes.append(
                            Change(
                                Action.MOVED_RESULT,
                                loc[0],
                                tid,
                                f"moved from messages[{loc[0]}] into the user message after messages[{i}]",
                            )
                        )
                else:
                    results.append(
                        {"type": "tool_result", "tool_use_id": tid, "content": placeholder, "is_error": True}
                    )
                    changes.append(
                        Change(
                            Action.SYNTHESIZED_RESULT,
                            i,
                            tid,
                            "added a placeholder tool_result (is_error=true) for a tool_use with no result",
                        )
                    )

            if not new_blocks:
                changes.append(
                    Change(
                        Action.DROPPED_MESSAGE,
                        i,
                        None,
                        "assistant message was left empty after dropping its tool_use blocks",
                    )
                )
                continue
            new_m = copy.deepcopy(m)
            new_m["content"] = new_blocks
            out.append(new_m)

            j = _next_user(messages, i)
            if j is not None:
                merged_users.add(j)
                nxt = messages[j]
                if isinstance(nxt.get("content"), list):
                    rest, _ = leftover(j)
                    original_first_bad = any(
                        _is_result(b) and any(not _is_result(x) for x in _blocks(nxt)[:bi])
                        for bi, b in enumerate(_blocks(nxt))
                    )
                    if original_first_bad and results:
                        changes.append(
                            Change(
                                Action.REORDERED_CONTENT,
                                j,
                                None,
                                "moved tool_result blocks ahead of other content",
                            )
                        )
                    content: Any = results + rest
                else:
                    text = nxt.get("content")
                    rest_blocks = [] if text in (None, "") else [{"type": "text", "text": text}]
                    content = (results + rest_blocks) if results else text
                if content in ([], None, ""):
                    changes.append(
                        Change(
                            Action.DROPPED_MESSAGE,
                            j,
                            None,
                            "user message was left empty after removing stray tool_result blocks",
                        )
                    )
                    continue
                new_u = copy.deepcopy(nxt)
                new_u["content"] = content
                out.append(new_u)
            elif results:
                out.append({"role": "user", "content": results})
            continue

        if role == "user" and isinstance(m.get("content"), list):
            rest, had = leftover(i)
            if had and not rest:
                changes.append(
                    Change(
                        Action.DROPPED_MESSAGE,
                        i,
                        None,
                        "user message was left empty after removing stray tool_result blocks",
                    )
                )
                continue
            new_u = copy.deepcopy(m)
            new_u["content"] = rest
            out.append(new_u)
            continue

        out.append(copy.deepcopy(m))
    return out, changes


def units(messages: List[Message]) -> List[List[int]]:
    """Atomic units: an assistant tool_use message plus the user message that
    carries its results; every other message is its own unit."""
    out: List[List[int]] = []
    i = 0
    while i < len(messages):
        j = _next_user(messages, i)
        if use_ids(messages[i]) and j is not None and any(_is_result(b) for b in _blocks(messages[j])):
            out.append([i, j])
            i = j + 1
        else:
            out.append([i])
            i += 1
    return out
