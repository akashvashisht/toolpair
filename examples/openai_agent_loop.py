"""Example 1: keep a long-running OpenAI agent's history valid and in budget.

Runs offline: `fake_model` stands in for the API call. Replace it with
`client.chat.completions.create(model=..., messages=messages, tools=...)`.
"""

import json

import toolpair

MAX_MESSAGES = 12


def fake_model(messages):
    # Pretend the model asks for a tool on every odd user turn.
    last_user = next(m for m in reversed(messages) if m["role"] == "user")
    turn = int(last_user["content"].split()[-1])
    if turn % 2:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{turn}",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": json.dumps({"q": turn})},
                }
            ],
        }
    return {"role": "assistant", "content": f"answer {turn}"}


history = [{"role": "system", "content": "You are a helpful agent."}]

for turn in range(1, 16):
    history.append({"role": "user", "content": f"question {turn}"})

    # 1) Fix any damage (e.g. a tool that crashed last turn), 2) fit the budget.
    request = toolpair.trim(history, max_messages=MAX_MESSAGES)
    assert toolpair.is_valid(request)

    reply = fake_model(request)
    history.append(reply)

    for call in reply.get("tool_calls", []):
        if turn == 7:
            continue  # simulate a tool that crashed: no result is recorded
        history.append({"role": "tool", "tool_call_id": call["id"], "content": "42"})

print("stored history issues:", [str(i) for i in toolpair.check(history)])
final = toolpair.trim(history, max_messages=MAX_MESSAGES)
print(f"sent {len(final)} of {len(history)} messages; valid = {toolpair.is_valid(final)}")
print("first kept roles:", [m["role"] for m in final[:4]])
