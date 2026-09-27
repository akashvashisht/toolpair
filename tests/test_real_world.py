"""Reproductions of failures reported in public issue trackers.

Each test rebuilds the message shape described in the linked report, checks
that toolpair detects the problem the provider rejects, and checks that the
repaired or trimmed output satisfies the provider's rule.
"""

import unittest

from helpers import a_res, a_results, a_use, a_user, o_asst, o_call, o_sys, o_tool, o_user

import toolpair as tp
from toolpair import IssueCode


class RealWorld(unittest.TestCase):
    def test_langchain_29637_naive_trim_orphans_tool_message(self):
        """langchain-ai/langchain#29637: trimming splits an AIMessage with
        tool_calls from its ToolMessage. A naive "last N" slice starting on a
        tool message is what the API rejects with "messages with role 'tool'
        must be a response to a preceeding message with 'tool_calls'"."""
        h = [
            o_sys(),
            o_user("weather?"),
            o_call("call_1"),
            o_tool("call_1", "sunny"),
            o_asst("It's sunny."),
            o_user("thanks"),
        ]
        naive = h[-3:]
        self.assertEqual(naive[0]["role"], "tool")
        self.assertEqual(tp.check(naive)[0].code, IssueCode.ORPHAN_RESULT)

        out = tp.trim(h, max_messages=4)
        self.assertEqual(tp.check(out), [])
        self.assertNotEqual(out[1]["role"], "tool")

    def test_pydantic_ai_4728_interrupted_stream_poisons_history(self):
        """pydantic/pydantic-ai#4728: a streaming timeout leaves a tool call
        with no result; "once a conversation is poisoned, it stays poisoned"."""
        h = [
            o_user("book a flight"),
            o_call("call_A", "call_B"),
            o_tool("call_A", "ok"),
            # stream timed out before call_B returned; user keeps chatting
            o_user("hello?"),
        ]
        self.assertEqual([i.code for i in tp.check(h)], [IssueCode.MISSING_RESULT])
        r = tp.repair(h)
        self.assertEqual(tp.check(r.messages), [])
        self.assertEqual(r.messages[3]["tool_call_id"], "call_B")

    def test_claude_code_3886_tool_use_without_result(self):
        """anthropics/claude-code#3886: "tool_use ids were found without
        tool_result blocks immediately after" after an interrupted tool."""
        h = [
            a_user("run the tests"),
            a_use("toolu_01", text="Running."),
            a_user("[Request interrupted by user] do something else"),
        ]
        self.assertEqual([i.code for i in tp.check(h)], [IssueCode.MISSING_RESULT])
        r = tp.repair(h)
        self.assertEqual(tp.check(r.messages), [])
        self.assertEqual(r.messages[2]["content"][0]["type"], "tool_result")

    def test_zoo_code_1307_images_interleaved_with_results(self):
        """Zoo-Code-Org/Zoo-Code#1307: tool results interleaved with image
        blocks break the Anthropic API."""
        img = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}
        h = [
            a_user("screenshot both"),
            a_use("A", "B"),
            {"role": "user", "content": [a_res("A"), img, a_res("B"), img]},
        ]
        self.assertTrue(tp.check(h))
        self.assertEqual(tp.check(tp.repair(h).messages), [])

    def test_openai_agents_1797_result_without_previous_tool_use(self):
        """openai/openai-agents-python#1797: "unexpected tool_use_id found in
        tool_result blocks ... must have a corresponding tool_use block in
        the previous message" when the assistant turn was flushed early."""
        h = [
            a_user("go"),
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "...", "signature": "s"}]},
            a_results("toolu_X"),
            a_use("toolu_X"),
        ]
        self.assertIn(IssueCode.MISPLACED_RESULT, [i.code for i in tp.check(h)])
        r = tp.repair(h)
        self.assertEqual(tp.check(r.messages), [])
        self.assertEqual(r.messages[-1]["content"][0]["tool_use_id"], "toolu_X")

    def test_autogen_7955_middle_pop_leaves_orphan(self):
        """microsoft/autogen#7955: a trimmer that pops messages from the
        middle leaves tool results whose call was removed."""
        full = [o_sys(), o_user("q"), o_call("c1"), o_tool("c1"), o_asst("a"), o_user("q2")]
        popped = [full[0], full[1], full[3], full[4], full[5]]  # call removed
        self.assertEqual([i.code for i in tp.check(popped)], [IssueCode.ORPHAN_RESULT])
        self.assertEqual(tp.check(tp.repair(popped).messages), [])


if __name__ == "__main__":
    unittest.main()
