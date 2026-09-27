import unittest

from helpers import a_results, a_use, a_user, o_asst, o_call, o_sys, o_tool, o_user

import toolpair as tp


class TrimOpenAI(unittest.TestCase):
    def setUp(self):
        self.h = [
            o_sys(),
            o_user("q1"),
            o_asst("a1"),
            o_user("q2"),
            o_call("a", "b"),
            o_tool("a"),
            o_tool("b"),
            o_asst("a2"),
            o_user("q3"),
            o_asst("a3"),
        ]

    def test_never_splits_unit(self):
        # Last 6 messages would naively start at tool("a"). The unit
        # [call, tool, tool] is 3 messages, so it can't fit; trim drops it whole.
        out = tp.trim(self.h, max_messages=6)
        self.assertEqual(tp.check(out), [])
        self.assertEqual(out[0]["role"], "system")
        self.assertNotIn("tool", [m["role"] for m in out])
        self.assertEqual(len(out), 4)

    def test_keeps_whole_unit_when_it_fits(self):
        out = tp.trim(self.h, max_messages=7)
        self.assertEqual(
            [m["role"] for m in out],
            ["system", "assistant", "tool", "tool", "assistant", "user", "assistant"],
        )
        self.assertEqual(tp.check(out), [])

    def test_system_not_kept_when_disabled(self):
        out = tp.trim(self.h, max_messages=2, keep_system=False)
        self.assertEqual(out, [o_user("q3"), o_asst("a3")])

    def test_start_on_user(self):
        out = tp.trim(self.h, max_messages=7, start_on_user=True)
        self.assertEqual([m["role"] for m in out], ["system", "user", "assistant"])

    def test_token_budget_with_custom_counter(self):
        out = tp.trim(self.h, max_tokens=5, token_counter=lambda m: 1)
        self.assertEqual(len(out), 4)
        self.assertEqual(tp.check(out), [])

    def test_default_token_counter(self):
        out = tp.trim(self.h, max_tokens=10_000)
        self.assertEqual(len(out), len(self.h))
        self.assertGreater(tp.approx_tokens(o_user("hello")), 0)

    def test_budget_too_small(self):
        with self.assertRaises(tp.BudgetTooSmallError):
            tp.trim(self.h, max_messages=1)

    def test_no_budget_given(self):
        with self.assertRaises(ValueError):
            tp.trim(self.h)
        with self.assertRaises(ValueError):
            tp.trim(self.h, max_messages=0)
        with self.assertRaises(ValueError):
            tp.trim(self.h, max_tokens=0)

    def test_repairs_before_trimming(self):
        broken = [o_user("q"), o_call("a"), o_user("next")]
        out = tp.trim(broken, max_messages=10)
        self.assertEqual(tp.check(out), [])

    def test_repair_first_false_rejects_broken_input(self):
        with self.assertRaises(ValueError):
            tp.trim([o_call("a")], max_messages=5, repair_first=False)

    def test_repair_first_false_returns_copies(self):
        h = [o_user("q"), o_asst("a")]
        out = tp.trim(h, max_messages=5, repair_first=False)
        out[0]["content"] = "changed"
        self.assertEqual(h[0]["content"], "q")

    def test_empty(self):
        self.assertEqual(tp.trim([], max_messages=3), [])


class TrimAnthropic(unittest.TestCase):
    def test_pair_kept_or_dropped_together(self):
        h = [a_user("q1"), a_use("t1"), a_results("t1"), a_use("t2"), a_results("t2")]
        out = tp.trim(h, max_messages=3)
        self.assertEqual(out, [a_use("t2"), a_results("t2")])
        self.assertEqual(tp.check(out), [])

    def test_start_on_user(self):
        h = [a_user("q1"), a_use("t1"), a_results("t1"), a_user("q2")]
        # With start_on_user the pair is dropped, leaving just the user turn.
        self.assertEqual(tp.trim(h, max_messages=3, start_on_user=True), [a_user("q2")])


if __name__ == "__main__":
    unittest.main()
