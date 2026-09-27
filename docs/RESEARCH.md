# Research, spec and review log

This document records how the problem was chosen, the spec the code was built
against, and an honest review of the result. Research date: 2026-09-27.

## 1. Constraints (set by the maintainer)

Python · AI/LLM tooling · weekend-sized v1 with light upkeep · MIT · no paid dependencies.

## 2. Candidate problems and evidence

About 19 candidates were collected from GitHub issues, Hacker News, dev blogs
and the OpenAI community forum. **Blocked sources:** Reddit, the Stack Exchange
API, and the Hacker News Algolia API were denied by the research environment's
network policy. GitHub reaction counts were mostly unavailable to the fetch
tool. The evidence therefore leans on GitHub issues, and "reach" scores are
judgment, not measurements.

**Dropped at Gate 1 (fewer than 3 independent sources):** per-model parameter quirks,
a context-window size registry, and tool arguments arriving as JSON strings.
**Dropped at Gate 2 (maintained solution exists):** streaming partial JSON
(jiter, partial-json-parser, jsonriver), record/replay tests (pytest-llm-vcr and
others), cost tracking with caching (pydantic/genai-prices), offline Claude
token counting (ctok), and Gemini token counting (SDK `LocalTokenizer`).

## 3. Scoring (1–5; Fit = weekend-feasible; Upkeep = low maintenance)

| Problem | Pain | Reach | Fit | Diff. | Upkeep | Total |
|---|---|---|---|---|---|---|
| **Tool call/result pairing: check, repair, trim** | 5 | 4 | 5 | 4 | 5 | **23** |
| Provider-compatible JSON Schemas from Pydantic | 5 | 5 | 3 | 3 | 2 | 18 |
| 429: rate limit vs. exhausted quota, honoring retry hints | 4 | 4 | 4 | 3 | 3 | 18 |
| Rebuilding parallel tool calls from a stream | 4 | 3 | 4 | 3 | 3 | 17 |
| Streaming `<think>` reasoning separation | 3 | 3 | 5 | 2 | 4 | 17 |
| MCP runtime outputSchema conformance | 3 | 3 | 3 | 3 | 3 | 15 |

**Chosen:** tool call/result pairing (confirmed by the maintainer at Gate 3).
The runner-up (schemas) had the biggest pain, but pydantic-ai already ships
importable transformers for it, and provider schema rules change often. That
conflicts with the light-upkeep constraint.

### Evidence for the chosen problem

| Source | Date | Status | Key point |
|---|---|---|---|
| [langchain#29637](https://github.com/langchain-ai/langchain/issues/29637) | 2025 | open | `trim_messages` splits AIMessage tool_calls from ToolMessages; a maintainer calls them an "atomic unit" |
| [pydantic-ai#4728](https://github.com/pydantic/pydantic-ai/issues/4728) | 2026-03-18 | closed (PR #6319) | "once a conversation is poisoned, it stays poisoned"; the reporter kept a 120-line sanitizer |
| [semantic-kernel#12708](https://github.com/microsoft/semantic-kernel/issues/12708) | 2025-07-12 | — | truncation ignores tool message boundaries |
| [autogen#7955](https://github.com/microsoft/autogen/issues/7955) | — | open | pops from the middle; checks only index 0 for orphans |
| agent-os#3170 | 2026-09-20 | — | "messages with role 'tool' must be a response to a preceeding message with 'tool_calls'" |
| [claude-code#3886](https://github.com/anthropics/claude-code/issues/3886) (plus #1894, #5662, #8233, #8897) | 2025 | — | "tool_use ids were found without tool_result blocks immediately after" |
| [Zoo-Code#1307](https://github.com/Zoo-Code-Org/Zoo-Code/issues/1307) | — | — | images interleaved between tool_result blocks |
| [openai-agents-python#1797](https://github.com/openai/openai-agents-python/issues/1797) | 2025-09-24 | closed | "unexpected `tool_use_id` found in `tool_result` blocks" |
| [hermes-agent#44394](https://github.com/NousResearch/hermes-agent/issues/44394) | 2026 | — | sanitizer only ran after compression, not before every request |

### Competition (Gate 2)

| Existing | What it covers | Gap |
|---|---|---|
| [LiteLLM message sanitization](https://docs.litellm.ai/docs/completion/message_sanitization) | adds dummy results, removes orphans | "only applied in the Anthropic message transformation pipeline"; requires LiteLLM; doesn't move misplaced results or trim |
| [agent-message-window](https://dev.to/mukundakatta/agent-message-window-slide-your-context-window-without-dropping-tool-calls-3aip) | pair-safe sliding window | Anthropic only; no repair; PyPI publish pending (2026-05-25) |
| Framework-internal fixes (pydantic-ai history processor, LangChain middleware, etc.) | inside each framework | not reusable outside that framework |

## 4. Spec

**Target user:** anyone who stores or trims chat history for tool-calling LLMs,
with or without a framework.

**Interface:** `check`, `is_valid`, `repair`, `trim`, `detect_format`, `approx_tokens`,
and the CLI (`toolpair check|repair|trim`).

**Rules encoded, each from a primary source:**
- OpenAI: the two 400 error texts quoted in `_openai.py`. They're read strictly: results must come *immediately* after the call.
- Anthropic: the tool-use docs (results must *immediately* follow; `tool_result` blocks go *first*) and the two error texts quoted in `_anthropic.py`.

**Success criteria and results**

| # | Criterion | Result |
|---|---|---|
| 1 | Repair output always passes `check()` **and** an independent oracle | ✅ 6,000 random broken histories × 2 strategies |
| 2 | `repair()` is idempotent and leaves valid input unchanged | ✅ property-tested |
| 3 | Inputs are never mutated | ✅ property-tested |
| 4 | Trim output is valid, fits the budget, and is an order-preserving subsequence | ✅ property-tested |
| 5 | Reproduces at least 2 real reported failures | ✅ 6 reproduced in `tests/test_real_world.py` |
| 6 | Zero runtime dependencies; Python 3.9+ | ✅ tests pass on 3.9, 3.10, 3.11, 3.12 and 3.13 |
| 7 | Repair of a 10k-message history in under 200 ms | ✅ about 57 ms (OpenAI, 12k) and 106–120 ms (Anthropic, 10k) on Python 3.11 |
| 8 | Core-logic coverage of 85% or more | ✅ 100% of library lines apart from the `__main__` guard (stdlib tracer) |

**Out of scope for v1:** Responses API, Gemini, LangChain/pydantic-ai objects,
role-alternation rules, and live API contract tests.

## 5. Assumption log

| # | Assumption / unknown | Resolution |
|---|---|---|
| 1 | Does OpenAI allow other messages between tool_calls and tool messages? | Not assumed. The strict reading is used, which is valid under either reading |
| 2 | Does Anthropic accept consecutive same-role messages, or a history that starts with an assistant message? | Avoided by design: synthesized results are merged into the next user message, and `start_on_user` is offered. Where a repair has to drop a message, a same-role run can remain; this is documented as a limitation |
| 3 | Does either API reject duplicate results for one call? | Unverified. Duplicates are flagged and removed as the conservative choice (the first result is kept) |
| 4 | No live API testing (no keys, no paid services) | Accepted by the maintainer. Tests cover the documented rules and the quoted real error messages |
| 5 | PyPI name `toolpair` availability | **UNVERIFIED**: PyPI was blocked by the build environment's network policy. Check before publishing |
| 6 | Wheel build and install | **UNVERIFIED locally** (Debian's setuptools patch breaks offline builds). CI's `build` job builds and smoke-tests the wheel |
| 7 | Reach/engagement numbers | Not measured (see §2). No numbers are claimed |

## 6. Self-review (as a skeptical senior engineer)

**Does it meet every success criterion?** Yes; see the table in §4, and the numbers came from actual runs.

**What could still be wrong**
- *Rules outside pairing* (role alternation, empty content, the `server_tool_use` + client-tool mixing rule Anthropic documents) aren't enforced. A repaired history can still be rejected for those reasons.
- *Moving a result backwards* across an interleaved user message changes the conversation's order. This is intended (it's the only valid layout), and it's recorded in `changes`.
- *Placeholder wording* shapes how the model reacts to a failed call. It can be configured, but the default is opinionated.
- *Malformed tool calls* (non-dict entries, missing ids) are passed through, not validated. `None` ids are treated as ordinary ids.
- *The token budget* uses a 4-characters-per-token heuristic by default, which is inaccurate for non-English text and images. It's documented; pass a real tokenizer for exact limits.

**Would the people who complained switch?**
- **Framework-free and custom-agent developers:** yes. That's exactly who this serves.
- **pydantic-ai users:** probably not. #4728 was closed with a built-in fix.
- **LangChain users:** only if they're willing to convert with `convert_to_openai_messages`. A native adapter is the most valuable roadmap item.
- **LiteLLM users on OpenAI-format models:** yes. LiteLLM's sanitizer doesn't cover that path.

## 7. Launch plan

- **Repo name:** `toolpair` (check GitHub and PyPI availability first; fallbacks: `toolpair-guard`, `pairsafe`)
- **Description:** "Check, repair and trim LLM chat histories without ever separating a tool call from its result. OpenAI + Anthropic, zero deps."
- **Topics:** `llm`, `openai`, `anthropic`, `claude`, `tool-calling`, `function-calling`, `ai-agents`, `context-window`, `python`
- **Announcement** (for Show HN, r/LocalLLaMA, and a comment on langchain#29637 / autogen#7955 where it's relevant and allowed):

  > **toolpair: stop "tool_use ids were found without tool_result blocks" errors for good.**
  > If you trim or persist agent histories, one interrupted tool call or bad cut
  > can break every later request. toolpair is a zero-dependency Python library
  > and CLI that checks, repairs (move / drop / placeholder, with an audit trail)
  > and trims OpenAI and Anthropic histories without ever splitting a tool call
  > from its result. It's property-tested against an independent implementation of
  > the provider rules. Feedback and failing histories welcome.
