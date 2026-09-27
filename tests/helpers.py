"""Small builders for test histories."""

from __future__ import annotations

from typing import Any, Dict, List


# ---- OpenAI Chat Completions --------------------------------------------
def o_user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def o_sys(text: str = "You are helpful.") -> Dict[str, Any]:
    return {"role": "system", "content": text}


def o_asst(text: str = "ok") -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


def o_call(*ids: str, content: Any = None) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": "f", "arguments": "{}"}} for i in ids
        ],
    }


def o_tool(tid: str, content: str = "result") -> Dict[str, Any]:
    return {"role": "tool", "tool_call_id": tid, "content": content}


# ---- Anthropic Messages -------------------------------------------------
def a_user(text: str) -> Dict[str, Any]:
    return {"role": "user", "content": text}


def a_asst(text: str = "ok") -> Dict[str, Any]:
    return {"role": "assistant", "content": text}


def a_use(*ids: str, text: str = "") -> Dict[str, Any]:
    blocks: List[Any] = [{"type": "text", "text": text}] if text else []
    blocks += [{"type": "tool_use", "id": i, "name": "f", "input": {}} for i in ids]
    return {"role": "assistant", "content": blocks}


def a_res(tid: str, content: str = "result") -> Dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tid, "content": content}


def a_results(*ids: str, extra: Any = None) -> Dict[str, Any]:
    blocks: List[Any] = [a_res(i) for i in ids]
    if extra:
        blocks += extra
    return {"role": "user", "content": blocks}
