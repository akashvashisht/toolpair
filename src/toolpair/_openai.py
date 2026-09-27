"""OpenAI Chat Completions format.

Rule enforced (from the API's own 400 errors):
  * "An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'."
  * "messages with role 'tool' must be a response to a preceeding message
    with 'tool_calls'."

toolpair reads "followed by" strictly: the tool messages must come
immediately after the assistant message, with nothing in between. Output that
meets the strict reading also meets any looser one.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Set, Tuple

from ._types import Action, Change, Issue, IssueCode, Message


def call_ids(m: Message) -> List[Optional[str]]:
    """Tool call ids of an assistant message, in order, de-duplicated."""
    if m.get("role") != "assistant":
        return []
    calls = m.get("tool_calls")
    if not isinstance(calls, list):
        return []
    out: List[Optional[str]] = []
    for tc in calls:
        if isinstance(tc, dict):
            tid = tc.get("id")
            if tid not in out:
                out.append(tid)
    return out


def _is_tool(m: Message) -> bool:
    return m.get("role") == "tool"


def _block_after(messages: List[Message], i: int) -> List[int]:
    """Indices of the contiguous run of tool messages right after index i."""
    out = []
    j = i + 1
    while j < len(messages) and _is_tool(messages[j]):
        out.append(j)
        j += 1
    return out


def check(messages: List[Message]) -> List[Issue]:
    issues: List[Issue] = []
    tool_ids: Dict[Optional[str], List[int]] = {}
    for k, m in enumerate(messages):
        if _is_tool(m):
            tool_ids.setdefault(m.get("tool_call_id"), []).append(k)

    all_call_ids: Set[Optional[str]] = set()
    claimed: Set[int] = set()
    answered: Set[Optional[str]] = set()

    # Pass 1: which tool messages sit where they belong.
    seen_by_call: Dict[int, Set[Optional[str]]] = {}
    for i, m in enumerate(messages):
        ids = call_ids(m)
        if not ids:
            continue
        all_call_ids.update(ids)
        seen: Set[Optional[str]] = set()
        for k in _block_after(messages, i):
            tid = messages[k].get("tool_call_id")
            if tid in ids and tid not in seen:
                seen.add(tid)
                claimed.add(k)
                answered.add(tid)
        seen_by_call[i] = seen

    # Pass 2: report calls whose result is not in place.
    for i, seen in seen_by_call.items():
        for tid in call_ids(messages[i]):
            if tid in seen:
                continue
            elsewhere = [k for k in tool_ids.get(tid, []) if k not in claimed]
            if elsewhere:
                issues.append(
                    Issue(
                        IssueCode.MISPLACED_RESULT,
                        i,
                        tid,
                        f"result is at messages[{elsewhere[0]}], not directly after the call",
                    )
                )
            else:
                issues.append(
                    Issue(
                        IssueCode.MISSING_RESULT,
                        i,
                        tid,
                        "tool call has no tool message answering it",
                    )
                )

    misplaced_seen: Set[Optional[str]] = set()
    for k, m in enumerate(messages):
        if not _is_tool(m) or k in claimed:
            continue
        tid = m.get("tool_call_id")
        if tid in answered:
            issues.append(
                Issue(
                    IssueCode.DUPLICATE_RESULT,
                    k,
                    tid,
                    "another tool message already answers this call",
                )
            )
        elif tid in all_call_ids:
            if tid in misplaced_seen:
                issues.append(
                    Issue(
                        IssueCode.DUPLICATE_RESULT,
                        k,
                        tid,
                        "another tool message already answers this call",
                    )
                )
            misplaced_seen.add(tid)  # first one is reported at the call site
        else:
            issues.append(
                Issue(
                    IssueCode.ORPHAN_RESULT,
                    k,
                    tid,
                    "tool message answers a call id that no assistant message made",
                )
            )
    return sorted(issues, key=lambda x: x.message_index)


def repair(messages: List[Message], strategy: str, placeholder: str) -> Tuple[List[Message], List[Change]]:
    tool_ids: Dict[Optional[str], List[int]] = {}
    for k, m in enumerate(messages):
        if _is_tool(m):
            tool_ids.setdefault(m.get("tool_call_id"), []).append(k)

    # Plan: pick exactly one result message for every (assistant, call id).
    used: Set[int] = set()
    assigned: Dict[Tuple[int, Optional[str]], int] = {}
    for i, m in enumerate(messages):
        ids = call_ids(m)
        if not ids:
            continue
        block = _block_after(messages, i)
        for tid in ids:
            cands = tool_ids.get(tid, [])
            pick: Optional[int] = next((k for k in cands if k in block and k not in used), None)
            if pick is None:
                pick = next((k for k in cands if k > i and k not in used), None)
            if pick is None:
                pick = next((k for k in reversed(cands) if k < i and k not in used), None)
            if pick is not None:
                used.add(pick)
                assigned[(i, tid)] = pick

    answered_ids = {tid for (_, tid) in assigned}
    out: List[Message] = []
    changes: List[Change] = []

    for i, m in enumerate(messages):
        if _is_tool(m):
            if i in used:
                continue  # emitted right after its call
            tid = m.get("tool_call_id")
            why = (
                "duplicate result for an already-answered call"
                if tid in answered_ids
                else "no matching tool call"
            )
            changes.append(Change(Action.DROPPED_RESULT, i, tid, why))
            continue

        ids = call_ids(m)
        if not ids:
            out.append(copy.deepcopy(m))
            continue

        block_set = set(_block_after(messages, i))
        new_m = copy.deepcopy(m)
        kept_calls = []
        results: List[Message] = []
        emitted: Set[Optional[str]] = set()
        for tc in new_m["tool_calls"]:
            tid = tc.get("id") if isinstance(tc, dict) else None
            src = assigned.get((i, tid))
            if src is not None:
                kept_calls.append(tc)
                if tid not in emitted:
                    results.append(copy.deepcopy(messages[src]))
                    emitted.add(tid)
                    if src not in block_set:
                        changes.append(
                            Change(
                                Action.MOVED_RESULT,
                                src,
                                tid,
                                f"moved from messages[{src}] to directly after messages[{i}]",
                            )
                        )
            elif strategy == "synthesize":
                kept_calls.append(tc)
                if tid not in emitted:
                    results.append({"role": "tool", "tool_call_id": tid, "content": placeholder})
                    emitted.add(tid)
                    changes.append(
                        Change(
                            Action.SYNTHESIZED_RESULT,
                            i,
                            tid,
                            "added a placeholder tool message for a call with no result",
                        )
                    )
            else:
                changes.append(
                    Change(
                        Action.DROPPED_CALL,
                        i,
                        tid,
                        "removed a tool call that had no result",
                    )
                )

        if kept_calls:
            new_m["tool_calls"] = kept_calls
            out.append(new_m)
            out.extend(results)
        else:
            del new_m["tool_calls"]
            if new_m.get("content") not in (None, "", []):
                out.append(new_m)
            else:
                changes.append(
                    Change(
                        Action.DROPPED_MESSAGE,
                        i,
                        None,
                        "assistant message was left empty after dropping its tool calls",
                    )
                )
    return out, changes


def units(messages: List[Message]) -> List[List[int]]:
    """Group indices into atomic units: an assistant tool-call message plus its
    tool messages form one unit; every other message is its own unit."""
    out: List[List[int]] = []
    i = 0
    while i < len(messages):
        if call_ids(messages[i]):
            block = _block_after(messages, i)
            out.append([i, *block])
            i = (block[-1] + 1) if block else i + 1
        else:
            out.append([i])
            i += 1
    return out
