import copy
import unittest

from helpers import a_asst, a_res, a_results, a_use, a_user

import toolpair as tp
from toolpair import Action, IssueCode

IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}


def codes(issues):
    return [(i.code, i.message_index, i.tool_call_id) for i in issues]


class CheckAnthropic(unittest.TestCase):
    def test_valid(self):
        h = [a_user("hi"), a_use("t1", "t2", text="checking"), a_results("t1", "t2"), a_asst()]
        self.assertEqual(tp.check(h), [])
        self.assertEqual(tp.detect_format(h), "anthropic")

    def test_text_after_results_is_valid(self):
        h = [a_user("hi"), a_use("t1"), a_results("t1", extra=[{"type": "text", "text": "more"}])]
        self.assertEqual(tp.check(h), [])

    def test_missing(self):
        h = [a_user("hi"), a_use("t1"), a_user("what happened?")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISSING_RESULT, 1, "t1")])

    def test_missing_at_end(self):
        h = [a_user("hi"), a_use("t1")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISSING_RESULT, 1, "t1")])

    def test_orphan(self):
        h = [a_user("hi"), a_asst(), a_results("ghost")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.ORPHAN_RESULT, 2, "ghost")])

    def test_misplaced_two_messages_later(self):
        h = [a_user("hi"), a_use("t1"), a_user("wait"), a_asst("sure"), a_results("t1")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.MISPLACED_RESULT, 1, "t1")])

    def test_duplicate(self):
        h = [a_user("hi"), a_use("t1"), a_results("t1", "t1")]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.DUPLICATE_RESULT, 2, "t1")])

    def test_result_not_first(self):
        h = [
            a_user("hi"),
            a_use("t1"),
            {"role": "user", "content": [{"type": "text", "text": "x"}, a_res("t1")]},
        ]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.RESULT_NOT_FIRST, 2, "t1")])

    def test_server_tool_blocks_ignored(self):
        h = [
            a_user("hi"),
            {
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}},
                    {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": []},
                    {"type": "text", "text": "found it"},
                ],
            },
        ]
        self.assertEqual(tp.check(h, format="anthropic"), [])


class RepairAnthropic(unittest.TestCase):
    def test_noop_on_valid(self):
        h = [a_user("hi"), a_use("t1"), a_results("t1"), a_asst()]
        r = tp.repair(h)
        self.assertFalse(r.changed)
        self.assertEqual(r.messages, h)

    def test_synthesize_merges_into_following_user_text(self):
        h = [a_user("hi"), a_use("t1"), a_user("what happened?")]
        r = tp.repair(h)
        self.assertEqual(len(r.messages), 3)
        content = r.messages[2]["content"]
        self.assertEqual(content[0]["type"], "tool_result")
        self.assertTrue(content[0]["is_error"])
        self.assertEqual(content[1], {"type": "text", "text": "what happened?"})
        self.assertEqual(tp.check(r.messages), [])

    def test_synthesize_at_end_appends_user_message(self):
        r = tp.repair([a_user("hi"), a_use("t1")])
        self.assertEqual(r.messages[-1]["role"], "user")
        self.assertEqual(r.messages[-1]["content"][0]["tool_use_id"], "t1")

    def test_synthesize_when_next_is_assistant(self):
        h = [a_user("hi"), a_use("t1"), a_asst("hmm")]
        r = tp.repair(h)
        self.assertEqual([m["role"] for m in r.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(tp.check(r.messages), [])

    def test_drop_strategy(self):
        h = [a_user("hi"), a_use("t1", text="let me look"), a_user("?")]
        r = tp.repair(h, strategy="drop")
        self.assertEqual(r.messages[1]["content"], [{"type": "text", "text": "let me look"}])
        self.assertEqual(r.messages[2], a_user("?"))

    def test_drop_strategy_empty_assistant_removed(self):
        r = tp.repair([a_user("hi"), a_use("t1"), a_user("?")], strategy="drop")
        self.assertEqual(r.messages, [a_user("hi"), a_user("?")])

    def test_move_misplaced(self):
        h = [a_user("hi"), a_use("t1"), a_user("wait"), a_asst("sure"), a_results("t1")]
        r = tp.repair(h)
        self.assertEqual(tp.check(r.messages), [])
        self.assertEqual(r.messages[2]["content"][0]["tool_use_id"], "t1")
        self.assertEqual(r.messages[2]["content"][0]["content"], "result")
        self.assertEqual(r.messages[2]["content"][1], {"type": "text", "text": "wait"})
        self.assertEqual(len(r.messages), 4)  # emptied message dropped
        self.assertIn(Action.MOVED_RESULT, [c.action for c in r.changes])

    def test_orphan_dropped_and_empty_message_removed(self):
        h = [a_user("hi"), a_asst(), a_results("ghost"), a_user("again")]
        r = tp.repair(h)
        self.assertEqual(r.messages, [a_user("hi"), a_asst(), a_user("again")])

    def test_orphan_dropped_but_text_kept(self):
        h = [a_user("hi"), a_asst(), a_results("ghost", extra=[{"type": "text", "text": "k"}])]
        r = tp.repair(h)
        self.assertEqual(r.messages[2]["content"], [{"type": "text", "text": "k"}])

    def test_reorder_results_first(self):
        h = [
            a_user("hi"),
            a_use("t1"),
            {"role": "user", "content": [{"type": "text", "text": "x"}, a_res("t1")]},
        ]
        r = tp.repair(h)
        self.assertEqual([b["type"] for b in r.messages[2]["content"]], ["tool_result", "text"])
        self.assertIn(Action.REORDERED_CONTENT, [c.action for c in r.changes])

    def test_does_not_mutate_input(self):
        h = [a_user("hi"), a_use("t1"), a_user("x"), a_results("zz")]
        before = copy.deepcopy(h)
        tp.repair(h)
        self.assertEqual(h, before)

    def test_consecutive_tool_use_turns(self):
        h = [a_user("go"), a_use("x"), a_use("y"), a_results("x", "y")]
        r = tp.repair(h)
        self.assertEqual(tp.check(r.messages), [])
        self.assertEqual([m["role"] for m in r.messages], ["user", "assistant", "user", "assistant", "user"])

    def test_repeated_tool_use_id_gets_one_result(self):
        h = [a_user("hi"), a_use("t1", "t1")]
        r = tp.repair(h)
        self.assertEqual(len(r.messages[2]["content"]), 1)
        self.assertEqual(tp.check(r.messages), [])

    def test_images_between_results(self):
        # Zoo-Code-Org/Zoo-Code#1307: [result A, image, result B, image, text]
        h = [
            a_user("shoot"),
            a_use("A", "B"),
            {"role": "user", "content": [a_res("A"), IMG, a_res("B"), IMG, {"type": "text", "text": "done"}]},
        ]
        self.assertEqual(codes(tp.check(h)), [(IssueCode.RESULT_NOT_FIRST, 2, "B")])
        r = tp.repair(h)
        self.assertEqual(
            [b["type"] for b in r.messages[2]["content"]],
            ["tool_result", "tool_result", "image", "image", "text"],
        )
        self.assertEqual(tp.check(r.messages), [])


class Format(unittest.TestCase):
    def test_mixed_formats_raise(self):
        h = [{"role": "tool", "tool_call_id": "a", "content": ""}, a_results("x")]
        with self.assertRaises(ValueError):
            tp.check(h)

    def test_plain_history_defaults_to_openai(self):
        self.assertEqual(tp.detect_format([a_user("hi")]), "openai")

    def test_bad_format_name(self):
        with self.assertRaises(ValueError):
            tp.check([], format="gemini")

    def test_bad_input_types(self):
        with self.assertRaises(TypeError):
            tp.check("not a list")
        with self.assertRaises(TypeError):
            tp.check([1])
        with self.assertRaises(ValueError):
            tp.check([{"content": "no role"}])


if __name__ == "__main__":
    unittest.main()
