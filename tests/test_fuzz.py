"""Randomised property tests (stdlib only, fixed seeds so failures reproduce).

Properties checked on thousands of random, deliberately broken histories:
  1. repair() output always passes check().
  2. repair() is idempotent: repairing its own output changes nothing.
  3. repair()/check()/trim() never mutate their input.
  4. trim() output always passes check(), respects the message budget, and
     is a subsequence of the repaired history (nothing invented or reordered).
  5. every message in the input that is not a tool message/result survives
     repair() unless the "drop" strategy emptied it.
"""

import copy
import random
import unittest

from helpers import a_asst, a_res, a_use, a_user, o_asst, o_call, o_sys, o_tool, o_user

import toolpair as tp

N_CASES = 3000


def random_openai(rng: random.Random):
    ids = [f"c{i}" for i in range(6)]
    h = []
    for _ in range(rng.randint(0, 14)):
        r = rng.random()
        if r < 0.1:
            h.append(o_sys())
        elif r < 0.3:
            h.append(o_user(f"u{rng.random():.3f}"))
        elif r < 0.45:
            h.append(o_asst(f"a{rng.random():.3f}"))
        elif r < 0.7:
            h.append(o_call(*rng.sample(ids, rng.randint(1, 3)), content=rng.choice([None, "", "thinking"])))
        else:
            h.append(o_tool(rng.choice(ids + ["ghost"]), f"r{rng.random():.3f}"))
    return h


def random_anthropic(rng: random.Random):
    ids = [f"t{i}" for i in range(6)]
    h = []
    for _ in range(rng.randint(0, 12)):
        r = rng.random()
        if r < 0.2:
            h.append(a_user(f"u{rng.random():.3f}"))
        elif r < 0.35:
            h.append(a_asst(f"a{rng.random():.3f}"))
        elif r < 0.6:
            h.append(a_use(*rng.sample(ids, rng.randint(1, 3)), text=rng.choice(["", "hm"])))
        else:
            blocks = []
            for _ in range(rng.randint(1, 4)):
                if rng.random() < 0.7:
                    blocks.append(a_res(rng.choice(ids + ["ghost"]), f"r{rng.random():.3f}"))
                else:
                    blocks.append({"type": "text", "text": "note"})
            h.append({"role": "user", "content": blocks})
    return h


def oracle_openai(h):
    """Independent, literal encoding of the documented OpenAI rules (written
    separately from the library so the two can cross-check each other)."""
    for i, m in enumerate(h):
        if m["role"] == "assistant" and m.get("tool_calls"):
            want = {tc["id"] for tc in m["tool_calls"]}
            got, j = set(), i + 1
            while j < len(h) and h[j]["role"] == "tool":
                got.add(h[j]["tool_call_id"])
                j += 1
            if not want <= got:
                return False
        if m["role"] == "tool":
            k = i - 1
            while k >= 0 and h[k]["role"] == "tool":
                k -= 1
            if k < 0 or not h[k].get("tool_calls"):
                return False
            if m["tool_call_id"] not in {tc["id"] for tc in h[k]["tool_calls"]}:
                return False
    return True


def oracle_anthropic(h):
    """Independent, literal encoding of the documented Anthropic rules."""

    def uses(m):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            return set()
        return {b["id"] for b in m["content"] if b["type"] == "tool_use"}

    for i, m in enumerate(h):
        want = uses(m)
        if want:
            if i + 1 >= len(h) or h[i + 1]["role"] != "user" or not isinstance(h[i + 1]["content"], list):
                return False
            got = {b["tool_use_id"] for b in h[i + 1]["content"] if b["type"] == "tool_result"}
            if not want <= got:
                return False
        if m["role"] == "user" and isinstance(m["content"], list):
            types = [b["type"] for b in m["content"]]
            if "tool_result" in types:
                n = types.count("tool_result")
                if types[:n] != ["tool_result"] * n:
                    return False  # results must come first
                prev = uses(h[i - 1]) if i > 0 else set()
                if any(b["tool_use_id"] not in prev for b in m["content"] if b["type"] == "tool_result"):
                    return False
    return True


ORACLES = {"openai": oracle_openai, "anthropic": oracle_anthropic}


def is_subsequence(small, big):
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


class Fuzz(unittest.TestCase):
    def _run(self, gen, fmt):
        rng = random.Random(1234 if fmt == "openai" else 5678)
        for case in range(N_CASES):
            h = gen(rng)
            before = copy.deepcopy(h)
            oracle = ORACLES[fmt]
            # check() must never miss a violation the independent oracle sees.
            if not oracle(h):
                self.assertTrue(tp.check(h, format=fmt), f"check missed a violation, case {case}: {h}")
            for strategy in ("synthesize", "drop"):
                r = tp.repair(h, format=fmt, strategy=strategy)
                self.assertEqual(tp.check(r.messages, format=fmt), [], f"case {case} {strategy}: {h}")
                self.assertTrue(oracle(r.messages), f"oracle rejects repair, case {case}: {h}")
                again = tp.repair(r.messages, format=fmt, strategy=strategy)
                self.assertFalse(again.changed, f"not idempotent, case {case}: {h}")
                # Changes happen exactly when there were issues to fix.
                self.assertEqual(bool(r.issues), r.changed, f"case {case}: {h}")
            budget = rng.randint(1, 12)
            try:
                out = tp.trim(h, format=fmt, max_messages=budget)
            except tp.BudgetTooSmallError:
                out = None
            if out is not None:
                self.assertEqual(tp.check(out, format=fmt), [], f"trim case {case}")
                self.assertTrue(oracle(out), f"oracle rejects trim, case {case}")
                self.assertLessEqual(len(out), budget)
                repaired = tp.repair(h, format=fmt).messages
                self.assertTrue(is_subsequence(out, repaired), f"trim reordered, case {case}")
            self.assertEqual(h, before, f"input mutated, case {case}")

    def test_openai(self):
        self._run(random_openai, "openai")

    def test_anthropic(self):
        self._run(random_anthropic, "anthropic")

    def test_plain_messages_survive_repair(self):
        rng = random.Random(99)
        for _ in range(500):
            h = random_openai(rng)
            out = tp.repair(h).messages
            plain_in = [m for m in h if m["role"] in ("user", "system")]
            plain_out = [m for m in out if m["role"] in ("user", "system")]
            self.assertEqual(plain_in, plain_out)


if __name__ == "__main__":
    unittest.main()
