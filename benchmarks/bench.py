"""Timing benchmark for check/repair/trim on large histories.

Run: python benchmarks/bench.py
Builds a realistic agent transcript (user turn, assistant with 1-3 parallel
tool calls, their results, assistant answer), then damages ~5% of tool
results (dropped or displaced) to exercise repair.
"""

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import toolpair as tp  # noqa: E402


def build(n_turns: int, fmt: str, damage: float, seed: int = 0):
    rng = random.Random(seed)
    h = []
    for t in range(n_turns):
        ids = [f"call_{t}_{k}" for k in range(rng.randint(1, 3))]
        if fmt == "openai":
            h.append({"role": "user", "content": f"question {t} " + "x" * 200})
            h.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": i,
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q":"x"}'},
                        }
                        for i in ids
                    ],
                }
            )
            h += [
                {"role": "tool", "tool_call_id": i, "content": "r" * 500}
                for i in ids
                if rng.random() > damage
            ]
            h.append({"role": "assistant", "content": "answer " + "y" * 300})
        else:
            h.append({"role": "user", "content": f"question {t} " + "x" * 200})
            h.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": i, "name": "search", "input": {"q": "x"}} for i in ids
                    ],
                }
            )
            h.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": i, "content": "r" * 500}
                        for i in ids
                        if rng.random() > damage
                    ],
                }
            )
            h.append({"role": "assistant", "content": "answer " + "y" * 300})
    return h


def timeit(fn, repeat=5):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


if __name__ == "__main__":
    print(f"Python {sys.version.split()[0]}  toolpair {tp.__version__}")
    print(f"{'format':10s} {'messages':>9s} {'check ms':>9s} {'repair ms':>10s} {'trim ms':>8s}")
    for fmt in ("openai", "anthropic"):
        for turns in (250, 2500):
            h = build(turns, fmt, damage=0.05)
            c = timeit(lambda h=h, fmt=fmt: tp.check(h, format=fmt))
            r = timeit(lambda h=h, fmt=fmt: tp.repair(h, format=fmt))
            tr = timeit(lambda h=h, fmt=fmt: tp.trim(h, format=fmt, max_tokens=100_000))
            print(f"{fmt:10s} {len(h):9d} {c:9.1f} {r:10.1f} {tr:8.1f}")
