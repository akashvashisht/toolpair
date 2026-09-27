"""Command-line interface.

    toolpair check   history.json
    toolpair repair  history.json -o fixed.json [--strategy drop]
    toolpair trim    history.json --max-messages 40 -o trimmed.json

FILE may be ``-`` for stdin. The JSON may be a list of messages or an object
with a ``"messages"`` key (other keys are preserved on output).

Exit codes: 0 = ok, 1 = issues found (check only), 2 = usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, List, Optional, Tuple

from . import __version__
from ._types import BudgetTooSmallError
from .api import check, repair, trim


def _load(path: str) -> Tuple[Any, List[Any]]:
    if path == "-":
        text = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    data = json.loads(text)
    if isinstance(data, list):
        return data, data
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        return data, data["messages"]
    raise ValueError("JSON must be a list of messages or an object with a 'messages' list")


def _dump(container: Any, messages: List[Any], out: Optional[str]) -> None:
    container = {**container, "messages": messages} if isinstance(container, dict) else messages
    text = json.dumps(container, indent=2, ensure_ascii=False) + "\n"
    if out is None or out == "-":
        sys.stdout.write(text)
    else:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="toolpair",
        description="Check, repair and trim LLM chat histories without breaking tool call/result pairs.",
    )
    p.add_argument("--version", action="version", version=f"toolpair {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("file", help="JSON file with the message history, or - for stdin")
        sp.add_argument("--format", default="auto", choices=["auto", "openai", "anthropic"])

    c = sub.add_parser("check", help="report pairing issues (exit 1 if any)")
    common(c)
    c.add_argument("--json", action="store_true", help="print issues as JSON")

    r = sub.add_parser("repair", help="fix pairing issues")
    common(r)
    r.add_argument("-o", "--output", help="output file (default: stdout)")
    r.add_argument("--strategy", default="synthesize", choices=["synthesize", "drop"])

    t = sub.add_parser("trim", help="keep recent messages within a budget, pairs intact")
    common(t)
    t.add_argument("-o", "--output", help="output file (default: stdout)")
    t.add_argument("--max-messages", type=int)
    t.add_argument("--max-tokens", type=int, help="approximate (4 chars/token) budget")
    t.add_argument("--start-on-user", action="store_true")

    args = p.parse_args(argv)
    try:
        container, messages = _load(args.file)
        if args.cmd == "check":
            issues = check(messages, format=args.format)
            if args.json:
                print(
                    json.dumps(
                        [
                            {
                                "code": i.code.value,
                                "message_index": i.message_index,
                                "tool_call_id": i.tool_call_id,
                                "detail": i.detail,
                            }
                            for i in issues
                        ],
                        indent=2,
                    )
                )
            else:
                for i in issues:
                    print(i)
                print(f"{len(issues)} issue(s) found" if issues else "OK: no pairing issues")
            return 1 if issues else 0
        if args.cmd == "repair":
            res = repair(messages, format=args.format, strategy=args.strategy)
            for ch in res.changes:
                print(ch, file=sys.stderr)
            print(f"{len(res.changes)} change(s) made", file=sys.stderr)
            _dump(container, res.messages, args.output)
            return 0
        out = trim(
            messages,
            format=args.format,
            max_messages=args.max_messages,
            max_tokens=args.max_tokens,
            start_on_user=args.start_on_user,
        )
        print(f"kept {len(out)} of {len(messages)} message(s)", file=sys.stderr)
        _dump(container, out, args.output)
        return 0
    except (OSError, ValueError, TypeError, BudgetTooSmallError) as e:
        print(f"toolpair: error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
