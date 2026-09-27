# toolpair

**Check, repair and trim LLM chat histories without ever separating a tool call from its result.**

Zero dependencies · Python 3.9+ · OpenAI Chat Completions and Anthropic Messages formats · MIT

```
400 An assistant message with 'tool_calls' must be followed by tool messages responding to each 'tool_call_id'.
400 messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.
400 `tool_use` ids were found without `tool_result` blocks immediately after: toolu_...
400 unexpected `tool_use_id` found in `tool_result` blocks: toolu_... Each `tool_result` block must have a corresponding `tool_use` block in the previous message.
```

If you've built an agent, you've probably seen at least one of these. They happen when:

- a **context-window trimmer** keeps the last N messages and the cut lands between a tool call and its result,
- a **stream times out or the user interrupts** while a tool is running, so the call is saved but its result never is,
- **persisted history** gets edited, merged or deserialized in the wrong order, or
- (Anthropic) **text or images** end up before the `tool_result` blocks in a user message.

Once one of these gets into a stored conversation, every later request fails. As one reporter put it, *"once a conversation is poisoned, it stays poisoned"* ([pydantic-ai#4728](https://github.com/pydantic/pydantic-ai/issues/4728)).

`toolpair` is a small, framework-free library that finds these problems, fixes them, and trims history safely.

## Install

```bash
pip install toolpair            # once published; until then:
pip install git+https://github.com/YOUR_GITHUB_USERNAME/toolpair
```

## Quick start

```python
import toolpair

history = [
    {"role": "user", "content": "Book me a flight"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_A", "type": "function", "function": {"name": "search", "arguments": "{}"}},
        {"id": "call_B", "type": "function", "function": {"name": "price", "arguments": "{}"}},
    ]},
    {"role": "tool", "tool_call_id": "call_A", "content": "3 flights found"},
    # the stream timed out before call_B returned...
    {"role": "user", "content": "hello?"},
]

for issue in toolpair.check(history):
    print(issue)
# messages[1] missing_result [call_B]: tool call has no tool message answering it

result = toolpair.repair(history)
for change in result.changes:
    print(change)
# messages[1] synthesized_result [call_B]: added a placeholder tool message for a call with no result

client.chat.completions.create(model=..., messages=result.messages)  # accepted
```

Trim to a budget without breaking pairs:

```python
recent = toolpair.trim(history, max_messages=40)                       # count
recent = toolpair.trim(history, max_tokens=100_000, token_counter=my_counter)  # tokens
```

It works the same way for Anthropic histories; the format is detected automatically:

```python
toolpair.repair(anthropic_messages).messages   # tool_result blocks moved first, orphans removed, ...
```

## What it does

### `check(messages, *, format="auto") -> list[Issue]`

Reports every pairing problem without changing anything. `is_valid(messages)` returns a bool.

| Code | Meaning |
|---|---|
| `missing_result` | A tool call has no result anywhere |
| `misplaced_result` | The result exists but isn't directly after the call (e.g. a user message slipped in between, or the result comes *before* the call) |
| `orphan_result` | A result references a call id that doesn't exist |
| `duplicate_result` | Two results answer the same call |
| `result_not_first` | (Anthropic) text or image blocks come before `tool_result` blocks |

### `repair(messages, *, format="auto", strategy="synthesize", placeholder=...) -> RepairResult`

| Problem | Fix |
|---|---|
| Misplaced result | **Moved** directly after its call (nothing is lost) |
| Orphan or duplicate result | **Dropped** (the first result is kept) |
| Missing result | `strategy="synthesize"` (default): adds a placeholder result telling the model the call failed (Anthropic: `is_error: true`). `strategy="drop"`: removes the call instead |
| Results not first (Anthropic) | `tool_result` blocks are moved ahead of other content, keeping order |
| A message left empty by a fix | Removed |

`RepairResult` has `.messages` (new list; your input is never mutated), `.changes` (an audit trail of every edit, with the original message index), and `.issues` (what `check()` found in the input). All other keys on your messages (`name`, `cache_control`, `refusal`, thinking blocks, etc.) are kept as they are.

### `trim(messages, *, max_messages=None, max_tokens=None, token_counter=None, keep_system=True, start_on_user=False, repair_first=True) -> list`

It groups messages into **atomic units**: an assistant tool call plus its results is one unit, and every other message is a unit of its own. It then keeps the newest units that fit and drops older ones whole. System/developer messages are pinned by default. If even the newest unit doesn't fit, it raises `BudgetTooSmallError` instead of silently returning an empty conversation.

`approx_tokens` (4 characters per token) is the default counter. It's an approximation, so pass your real tokenizer for exact limits.

### CLI

```bash
toolpair check  conversation.json            # exit 1 if issues found (CI-friendly)
toolpair check  conversation.json --json
toolpair repair conversation.json -o fixed.json [--strategy drop]
toolpair trim   conversation.json --max-messages 40 -o trimmed.json
cat conversation.json | toolpair check -
```

The file can be a JSON list of messages or an object with a `"messages"` key (other keys are preserved).

## How it compares

| | toolpair | LiteLLM message sanitization | LangChain `trim_messages` | agent-message-window | Hand-rolled per project |
|---|---|---|---|---|---|
| Works without adopting a framework | ✅ | ❌ requires LiteLLM | ❌ LangChain messages | ✅ | ✅ |
| OpenAI Chat Completions format | ✅ | ❌ applied to Anthropic calls only | via LangChain | ❌ | varies |
| Anthropic Messages format | ✅ | ✅ | via LangChain | ✅ | varies |
| Detects & reports issues | ✅ | debug logs only | ❌ | ❌ | rarely |
| Repairs damaged history (move / drop / synthesize) | ✅ | drop + synthesize (no move) | ❌ | ❌ | varies |
| Trims without splitting pairs | ✅ | ❌ | known issue ([#29637](https://github.com/langchain-ai/langchain/issues/29637)) | ✅ | varies |
| Audit trail of changes | ✅ | ❌ | ❌ | ❌ | rarely |
| Zero dependencies | ✅ | ❌ | ❌ | ✅ | ✅ |

This table is based on each project's public docs and issues as of September 2026. [LiteLLM's docs](https://docs.litellm.ai/docs/completion/message_sanitization) say its sanitizer is *"currently only applied in the Anthropic message transformation pipeline"* and is enabled with `modify_params=True`. [agent-message-window](https://dev.to/mukundakatta/agent-message-window-slide-your-context-window-without-dropping-tool-calls-3aip) supports the Anthropic format only and hadn't been published to PyPI at the time of writing. Corrections are welcome.

## Why this exists: the evidence

This problem has been reported independently across frameworks, again and again:

- [langchain#29637](https://github.com/langchain-ai/langchain/issues/29637): `trim_messages` separates tool calls from their results (still open)
- [pydantic-ai#4728](https://github.com/pydantic/pydantic-ai/issues/4728): the reporter kept a 120-line sanitizer of their own in production
- [semantic-kernel#12708](https://github.com/microsoft/semantic-kernel/issues/12708): a truncation reducer ignores tool-message boundaries
- [autogen#7955](https://github.com/microsoft/autogen/issues/7955): popping messages from the middle leaves orphaned results
- [claude-code#3886](https://github.com/anthropics/claude-code/issues/3886): `tool_use` ids without `tool_result` blocks
- [Zoo-Code#1307](https://github.com/Zoo-Code-Org/Zoo-Code/issues/1307): images interleaved with tool results
- [openai-agents-python#1797](https://github.com/openai/openai-agents-python/issues/1797): a `tool_result` with no `tool_use` in the previous message

Each of these is reproduced as a test in [`tests/test_real_world.py`](tests/test_real_world.py).

## How it's verified

- **Unit tests** for every issue type and every repair action, in both formats.
- **Reproductions** of the public reports listed above.
- **Property tests**: 3,000 random, deliberately broken histories per format, each checked by an *independent, literal re-implementation of the provider rules* (written separately from the library). They assert that repair output is always accepted, that repair is idempotent, that valid input is never changed, that the input is never mutated, and that trim output is always valid, fits the budget, and is an order-preserving subsequence.
- **Coverage**: 100% of library lines apart from the `__main__` guard.
- **Performance** (`benchmarks/bench.py`, Python 3.11): about 57 ms to repair a 12,000-message OpenAI history and about 120 ms for a 10,000-message Anthropic history. Most of that time goes to copying the messages.

## Limitations (v1)

- **Formats:** OpenAI Chat Completions and Anthropic Messages as plain dicts. SDK objects must be converted first (`msg.model_dump()`). The OpenAI Responses API, Gemini `contents`, LangChain message objects and the legacy `function_call` field aren't supported yet.
- **Pairing only:** it doesn't enforce rules that have nothing to do with tools, such as role alternation or whether the first message must be `user`. Use `start_on_user=True` when trimming if your provider needs it.
- **Strict reading of OpenAI's rule:** tool messages must come *immediately* after the assistant message. Output that meets this also meets any looser reading.
- **Anthropic server tools** (`server_tool_use`, `web_search_tool_result`, …) are left alone, because they're paired inside the assistant message.
- **No live API calls:** the tests check against the documented rules and the error messages quoted above, not against the live APIs.

## Roadmap

- [ ] OpenAI Responses API (`function_call` / `function_call_output` items)
- [ ] Gemini `contents` format (`functionCall` / `functionResponse`)
- [ ] Optional adapters for LangChain and pydantic-ai message types
- [ ] `summarize=` hook in `trim()` to compress dropped units instead of discarding them
- [ ] Optional live contract tests against provider APIs (opt-in, bring your own key)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports that include a (redacted) failing history are the most useful.

## License

MIT. See [LICENSE](LICENSE).
