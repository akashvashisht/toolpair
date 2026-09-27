# Contributing

Thanks for helping! This project is deliberately small: zero runtime
dependencies, one job done well.

## Setup

```bash
git clone https://github.com/akashvashisht/toolpair && cd toolpair
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check . && ruff format --check .
mypy
pytest --cov=toolpair --cov-fail-under=95
```

The same checks run in CI on Python 3.9 to 3.13.

## Guidelines

- **Rules come from provider docs or real API errors.** When adding or changing a
  rule, link the documentation or quote the error message in the code and PR.
  Don't encode guesses.
- **Every bug fix gets a regression test.** If it came from a real report, add
  it to `tests/test_real_world.py` with a link.
- **Keep the property tests green.** `tests/test_fuzz.py` checks repair and trim
  against an independent oracle. If you add a format, add an oracle for it.
- **Never mutate the caller's input** and never drop data silently. Every
  change must show up in `RepairResult.changes`.
- No new runtime dependencies.

## Reporting bugs

Include a **redacted** history that reproduces the problem (keep roles, ids and
block types; replace text with `"..."`), the output of `toolpair check`, and
the provider error you got.
