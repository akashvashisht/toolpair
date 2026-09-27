import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from helpers import o_call, o_tool, o_user

from toolpair.cli import main


def run(argv, stdin=None):
    out, err = io.StringIO(), io.StringIO()
    old_stdin = sys.stdin
    if stdin is not None:
        sys.stdin = io.StringIO(stdin)
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
    finally:
        sys.stdin = old_stdin
    return code, out.getvalue(), err.getvalue()


class CLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.broken = [o_user("hi"), o_call("a"), o_user("next")]
        self.path = os.path.join(self.tmp.name, "h.json")
        with open(self.path, "w") as f:
            json.dump(self.broken, f)

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_reports_and_exits_1(self):
        code, out, _ = run(["check", self.path])
        self.assertEqual(code, 1)
        self.assertIn("missing_result", out)

    def test_check_json(self):
        code, out, _ = run(["check", self.path, "--json"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)[0]["code"], "missing_result")

    def test_check_ok(self):
        code, out, _ = run(["check", "-"], stdin=json.dumps([o_user("x")]))
        self.assertEqual(code, 0)
        self.assertIn("OK", out)

    def test_repair_to_file_then_check(self):
        dest = os.path.join(self.tmp.name, "fixed.json")
        code, _, err = run(["repair", self.path, "-o", dest])
        self.assertEqual(code, 0)
        self.assertIn("1 change(s)", err)
        self.assertEqual(run(["check", dest])[0], 0)

    def test_repair_preserves_wrapper_object(self):
        data = {"id": "conv-1", "messages": self.broken}
        code, out, _ = run(["repair", "-", "--strategy", "drop"], stdin=json.dumps(data))
        self.assertEqual(code, 0)
        res = json.loads(out)
        self.assertEqual(res["id"], "conv-1")
        self.assertEqual(len(res["messages"]), 2)

    def test_trim(self):
        h = [o_user("q"), o_call("a"), o_tool("a"), o_user("q2")]
        code, out, err = run(["trim", "-", "--max-messages", "2"], stdin=json.dumps(h))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), [o_user("q2")])
        self.assertIn("kept 1 of 4", err)

    def test_errors_exit_2(self):
        self.assertEqual(run(["check", "/nonexistent.json"])[0], 2)
        self.assertEqual(run(["check", "-"], stdin="not json")[0], 2)
        self.assertEqual(run(["check", "-"], stdin='{"x": 1}')[0], 2)
        self.assertEqual(run(["trim", self.path])[0], 2)  # no budget


if __name__ == "__main__":
    unittest.main()
