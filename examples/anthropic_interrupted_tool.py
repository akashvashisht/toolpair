"""Example 2: recover an Anthropic conversation after the user interrupts a tool.

Without repair, the next request fails with:
  400 `tool_use` ids were found without `tool_result` blocks immediately after
"""

import json

import toolpair

stored = [
    {"role": "user", "content": "Take screenshots of both dashboards."},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Capturing both."},
            {"type": "tool_use", "id": "toolu_A", "name": "screenshot", "input": {"url": "a"}},
            {"type": "tool_use", "id": "toolu_B", "name": "screenshot", "input": {"url": "b"}},
        ],
    },
    # A's result was saved, then the user interrupted before B finished.
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_A", "content": "saved a.png"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."}},
            {"type": "text", "text": "Stop, just summarize what you have."},
        ],
    },
]

print("Issues:")
for issue in toolpair.check(stored):
    print("  ", issue)

result = toolpair.repair(stored)
print("\nChanges:")
for change in result.changes:
    print("  ", change)

print("\nRepaired user turn block order:", [b["type"] for b in result.messages[2]["content"]])
print("valid:", toolpair.is_valid(result.messages))
# client.messages.create(model=..., max_tokens=1024, messages=result.messages)
print(json.dumps(result.messages[2]["content"][1], indent=2))
