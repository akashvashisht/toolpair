"""Line coverage for src/toolpair using only the standard library.

Used where `coverage.py` is unavailable. CI uses pytest-cov instead.
Usage: python scripts/coverage_stdlib.py
"""

import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "toolpair")
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "tests")]

hits = set()


def tracer(frame, event, arg):
    fn = frame.f_code.co_filename
    if fn.startswith(SRC):
        if event == "line":
            hits.add((fn, frame.f_lineno))
        return tracer
    return None


def read(path):
    with open(path) as f:
        return f.read()


def executable_lines(path):
    tree = ast.parse(read(path))
    lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
        ):
            if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                continue  # docstrings
            lines.add(node.lineno)
    return lines


sys.settrace(tracer)
tests_dir = os.path.join(ROOT, "tests")
suite = unittest.defaultTestLoader.discover(tests_dir, top_level_dir=tests_dir)
result = unittest.TextTestRunner(verbosity=0).run(suite)
sys.settrace(None)

total = covered = 0
for name in sorted(os.listdir(SRC)):
    if not name.endswith(".py") or name == "__main__.py":
        continue
    path = os.path.join(SRC, name)
    src_lines = read(path).splitlines()
    ex = {n for n in executable_lines(path) if "pragma: no cover" not in src_lines[n - 1]}
    # module-level lines run at import time (before tracing); count them as hit
    tree = ast.parse(read(path))
    module_level = {n.lineno for n in tree.body}
    hit = {ln for (f, ln) in hits if f == path} | module_level
    miss = sorted(ex - hit)
    total += len(ex)
    covered += len(ex) - len(miss)
    print(f"{name:16s} {100 * (len(ex) - len(miss)) / max(len(ex), 1):6.1f}%  missing: {miss}")
print(f"TOTAL {100 * covered / total:.1f}%")
sys.exit(0 if result.wasSuccessful() else 1)
