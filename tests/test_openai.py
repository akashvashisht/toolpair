import copy
import unittest

from helpers import o_asst, o_call, o_sys, o_tool, o_user

import toolpair as tp
from toolpair import Action, IssueCode


def codes(issues):
    return [(i.code, i.message_index, i.tool_call_id) for i in issues]


class CheckOpenAI(unittest.TestCase):
    def test_valid_history(self):
        h = [o_sys(), o_user("hi"), o_call("a", "b"), o_tool("a"), o_tool("b"), o_asst()]
        self.assertEqual(tp.check(h), [])
        self.assertTrue(tp.is_valid(h))

    def test_results_in_any_order_within_block_are_valid(self):
        h = [o_user("hi"), o_call("a", "b"), o_tool("b"), o_tool("a")]
        self.assertEqual(tp.check(h), [])

    def test_missing_result(self):
        h = [o_user("hi"), o_call("a"), o_user("next")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISSING_RESULT, 1, "a")])

    def test_missing_at_end(self):
        h = [o_user("hi"), o_call("a", "b"), o_tool("a")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISSING_RESULT, 1, "b")])

    def test_orphan_result(self):
        h = [o_user("hi"), o_tool("x")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.ORPHAN_RESULT, 1, "x")])

    def test_misplaced_result_after_interleaved_message(self):
        h = [o_user("hi"), o_call("a"), o_user("interrupt"), o_tool("a")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISPLACED_RESULT, 1, "a")])

    def test_result_before_call(self):
        h = [o_user("hi"), o_tool("a"), o_call("a")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISPLACED_RESULT, 2, "a")])

    def test_duplicate_result(self):
        h = [o_user("hi"), o_call("a"), o_tool("a"), o_tool("a")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.DUPLICATE_RESULT, 3, "a")])

    def test_duplicate_misplaced_results(self):
        h = [o_user("hi"), o_call("a"), o_user("x"), o_tool("a"), o_tool("a")]
        got = codes(tp.check(h))
        self.assertIn((IssueCode.MISPLACED_RESULT, 1, "a"), got)
        self.assertIn((IssueCode.DUPLICATE_RESULT, 4, "a"), got)

    def test_empty_history(self):
        self.assertEqual(tp.check([]), [])


class RepairOpenAI(unittest.TestCase):
    def test_noop_on_valid(self):
        h = [o_user("hi"), o_call("a"), o_tool("a"), o_asst()]
        r = tp.repair(h)
        self.assertFalse(r.changed)
        self.assertEqual(r.messages, h)
        self.assertIsNot(r.messages[0], h[0])

    def test_synthesize_missing(self):
        h = [o_user("hi"), o_call("a"), o_user("next")]
        r = tp.repair(h)
        self.assertEqual([m["role"] for m in r.messages], ["user", "assistant", "tool", "user"])
        self.assertEqual(r.messages[2]["tool_call_id"], "a")
        self.assertEqual(r.messages[2]["content"], tp.DEFAULT_PLACEHOLDER)
        self.assertEqual([c.action for c in r.changes], [Action.SYNTHESIZED_RESULT])
        self.assertEqual(tp.check(r.messages), [])

    def test_custom_placeholder(self):
        r = tp.repair([o_call("a")], placeholder="CANCELLED")
        self.assertEqual(r.messages[1]["content"], "CANCELLED")

    def test_drop_missing_call_keeps_others(self):
        h = [o_user("hi"), o_call("a", "b"), o_tool("a")]
        r = tp.repair(h, strategy="drop")
        self.assertEqual([tc["id"] for tc in r.messages[1]["tool_calls"]], ["a"])
        self.assertEqual(len(r.messages), 3)
        self.assertEqual(tp.check(r.messages), [])

    def test_drop_all_calls_removes_empty_message(self):
        h = [o_user("hi"), o_call("a"), o_user("next")]
        r = tp.repair(h, strategy="drop")
        self.assertEqual(r.messages, [o_user("hi"), o_user("next")])
        self.assertIn(Action.DROPPED_MESSAGE, [c.action for c in r.changes])

    def test_drop_all_calls_keeps_message_with_text(self):
        h = [o_user("hi"), o_call("a", content="Let me check."), o_user("next")]
        r = tp.repair(h, strategy="drop")
        self.assertEqual(r.messages[1], {"role": "assistant", "content": "Let me check."})

    def test_move_misplaced(self):
        h = [o_user("hi"), o_call("a"), o_user("interrupt"), o_tool("a", "42")]
        r = tp.repair(h)
        self.assertEqual([m["role"] for m in r.messages], ["user", "assistant", "tool", "user"])
        self.assertEqual(r.messages[2]["content"], "42")
        self.assertEqual([c.action for c in r.changes], [Action.MOVED_RESULT])

    def test_move_result_that_came_before_call(self):
        h = [o_user("hi"), o_tool("a", "early"), o_call("a")]
        r = tp.repair(h)
        self.assertEqual([m["role"] for m in r.messages], ["user", "assistant", "tool"])
        self.assertEqual(r.messages[2]["content"], "early")

    def test_drop_orphan_and_duplicate(self):
        h = [o_user("hi"), o_tool("zzz"), o_call("a"), o_tool("a", "1"), o_tool("a", "2")]
        r = tp.repair(h)
        self.assertEqual(len(r.messages), 3)
        self.assertEqual(r.messages[2]["content"], "1")
        drops = [c for c in r.changes if c.action is Action.DROPPED_RESULT]
        self.assertEqual({c.message_index for c in drops}, {1, 4})

    def test_results_emitted_in_call_order(self):
        h = [o_call("a", "b"), o_tool("b", "B"), o_user("x"), o_tool("a", "A")]
        r = tp.repair(h)
        self.assertEqual([m.get("tool_call_id") for m in r.messages[1:3]], ["a", "b"])

    def test_does_not_mutate_input(self):
        h = [o_user("hi"), o_call("a"), o_user("x"), o_tool("zz")]
        before = copy.deepcopy(h)
        tp.repair(h)
        tp.check(h)
        self.assertEqual(h, before)

    def test_preserves_extra_keys(self):
        call = o_call("a")
        call["name"] = "bot"
        call["refusal"] = None
        r = tp.repair([o_user("hi"), call, o_tool("a")])
        self.assertEqual(r.messages[1]["name"], "bot")
        self.assertIn("refusal", r.messages[1])

    def test_issues_reported_in_result(self):
        r = tp.repair([o_call("a")])
        self.assertEqual(r.issues[0].code, IssueCode.MISSING_RESULT)

    def test_bad_strategy(self):
        with self.assertRaises(ValueError):
            tp.repair([], strategy="nope")


if __name__ == "__main__":
    unittest.main()
