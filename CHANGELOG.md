# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-09-27

### Added
- `check()` / `is_valid()`: detect missing, misplaced, orphan and duplicate tool
  results, and (Anthropic) results that don't come first.
- `repair()`: move misplaced results, drop orphans and duplicates, and synthesize
  or drop calls with no result. Returns an audit trail of every change.
- `trim()`: trim history to a message or token budget without splitting a
  tool call from its results.
- OpenAI Chat Completions and Anthropic Messages formats, with auto-detection.
- `toolpair` CLI with `check`, `repair` and `trim` commands.
