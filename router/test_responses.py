"""
test_responses.py - Responses API <-> chat completions translation. Run: pytest -q

Fixtures mirror what Codex sends (instructions + input items + function/custom
tools) and what its SSE parser reads back.
"""

import json

from responses import ResponseBuilder, responses_to_chat

SHELL_TOOL = {"type": "function", "name": "shell", "description": "Run a command",
              "strict": False,
              "parameters": {"type": "object", "properties": {"command": {"type": "array"}},
                             "required": ["command"]}}
PATCH_TOOL = {"type": "custom", "name": "apply_patch", "description": "Apply a patch",
              "format": {"type": "grammar", "syntax": "lark", "definition": "start: 'x'"}}


def events(gen):
    """Parse 'event: x\\ndata: {...}\\n\\n' strings into dicts."""
    out = []
    for raw in gen:
        for block in raw.strip().split("\n\n"):
            data = [l for l in block.split("\n") if l.startswith("data:")][0]
            out.append(json.loads(data[5:]))
    return out


# ------------------------------------------------------------ request -> chat

def test_instructions_become_the_system_message():
    chat, _ = responses_to_chat({"instructions": "be terse", "input": "hi"})
    assert chat["messages"][0] == {"role": "system", "content": "be terse"}
    assert chat["messages"][1] == {"role": "user", "content": "hi"}


def test_message_items_with_text_parts_and_developer_role():
    chat, _ = responses_to_chat({"input": [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"},
                                                        {"type": "input_text", "text": "b"}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
    ]})
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert chat["messages"][1]["content"] == "a\nb"


def test_function_call_history_round_trips_to_tool_messages():
    chat, _ = responses_to_chat({"input": [
        {"type": "message", "role": "user", "content": "list files"},
        {"type": "function_call", "call_id": "call_1", "name": "shell", "arguments": '{"command":["ls"]}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "a.py\nb.py"},
    ]})
    assistant, tool = chat["messages"][1], chat["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"] is None
    assert assistant["tool_calls"][0] == {"id": "call_1", "type": "function",
                                          "function": {"name": "shell", "arguments": '{"command":["ls"]}'}}
    assert tool == {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"}


def test_consecutive_function_calls_share_one_assistant_message():
    chat, _ = responses_to_chat({"input": [
        {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
        {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "x"},
        {"type": "function_call_output", "call_id": "c2", "output": "y"},
    ]})
    assert [m["role"] for m in chat["messages"]] == ["assistant", "tool", "tool"]
    assert len(chat["messages"][0]["tool_calls"]) == 2


def test_function_call_output_content_items_are_flattened():
    chat, _ = responses_to_chat({"input": [
        {"type": "function_call_output", "call_id": "c1",
         "output": [{"type": "input_text", "text": "line1"}, {"type": "input_text", "text": "line2"}]}]})
    assert chat["messages"][0]["content"] == "line1\nline2"


def test_reasoning_and_web_search_items_are_dropped():
    chat, _ = responses_to_chat({"input": [
        {"type": "reasoning", "summary": []},
        {"type": "web_search_call", "status": "completed"},
        {"type": "message", "role": "user", "content": "hi"}]})
    assert len(chat["messages"]) == 1


def test_function_tool_translates_verbatim():
    chat, custom = responses_to_chat({"input": "hi", "tools": [SHELL_TOOL]})
    assert chat["tools"] == [{"type": "function", "function": {
        "name": "shell", "description": "Run a command", "parameters": SHELL_TOOL["parameters"]}}]
    assert custom == {}


def test_custom_tool_becomes_a_string_input_function():
    chat, custom = responses_to_chat({"input": "hi", "tools": [PATCH_TOOL]})
    fn = chat["tools"][0]["function"]
    assert fn["name"] == "apply_patch"
    assert fn["parameters"]["required"] == ["input"]
    assert "start: 'x'" in fn["description"]
    assert "apply_patch" in custom


def test_custom_tool_call_history_is_replayed_as_input_argument():
    chat, _ = responses_to_chat({"input": [
        {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": "*** Begin Patch"},
        {"type": "custom_tool_call_output", "call_id": "c1", "output": "Done"}]})
    args = json.loads(chat["messages"][0]["tool_calls"][0]["function"]["arguments"])
    assert args == {"input": "*** Begin Patch"}


def test_generation_knobs_map_and_unknown_fields_are_ignored():
    chat, _ = responses_to_chat({"input": "hi", "max_output_tokens": 50, "temperature": 0.2,
                                 "store": False, "include": ["reasoning.encrypted_content"],
                                 "reasoning": {"effort": "high"}, "tools": [SHELL_TOOL],
                                 "tool_choice": "auto", "parallel_tool_calls": False, "stream": True})
    assert chat["max_tokens"] == 50 and chat["temperature"] == 0.2 and chat["stream"] is True
    assert chat["tool_choice"] == "auto" and chat["parallel_tool_calls"] is False
    assert "store" not in chat and "include" not in chat and "reasoning" not in chat


def test_unsupported_tool_types_are_dropped_not_fatal():
    chat, _ = responses_to_chat({"input": "hi", "tools": [{"type": "web_search"}, {"type": "namespace"}]})
    assert "tools" not in chat


# ------------------------------------------------------------ chat -> events

def chunk(delta, finish=None):
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def test_text_stream_produces_the_event_sequence_codex_reads():
    b = ResponseBuilder("m")
    evs = events(b.start())
    for c in (chunk({"role": "assistant"}), chunk({"content": "Hel"}), chunk({"content": "lo"}), chunk({}, "stop"),
              {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}):
        evs += events(b.feed(c))
    evs += events(b.finish())
    kinds = [e["type"] for e in evs]
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert kinds.count("response.output_text.delta") == 2
    assert "response.output_item.added" in kinds and "response.output_item.done" in kinds
    assert kinds[-1] == "response.completed"

    done = next(e for e in evs if e["type"] == "response.output_item.done")
    assert done["item"]["type"] == "message" and done["item"]["role"] == "assistant"
    assert done["item"]["content"] == [{"type": "output_text", "text": "Hello", "annotations": []}]

    completed = evs[-1]["response"]
    assert completed["status"] == "completed" and completed["id"].startswith("resp_")
    assert completed["usage"] == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9,
                                  "input_tokens_details": {"cached_tokens": 0},
                                  "output_tokens_details": {"reasoning_tokens": 0}}
    assert completed["output"] == [done["item"]]
    assert [e["sequence_number"] for e in evs] == list(range(1, len(evs) + 1))


def test_streamed_tool_call_fragments_assemble_into_a_function_call_item():
    b = ResponseBuilder("m")
    list(b.start())
    for c in (chunk({"tool_calls": [{"index": 0, "id": "call_9", "type": "function",
                                     "function": {"name": "shell", "arguments": ""}}]}),
              chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"command":'}}]}),
              chunk({"tool_calls": [{"index": 0, "function": {"arguments": '["ls"]}'}}]}),
              chunk({}, "tool_calls")):
        list(b.feed(c))
    evs = events(b.finish())
    done = [e for e in evs if e["type"] == "response.output_item.done"]
    assert len(done) == 1
    item = done[0]["item"]
    assert item == {"id": item["id"], "type": "function_call", "status": "completed",
                    "call_id": "call_9", "name": "shell", "arguments": '{"command":["ls"]}'}
    assert item["id"].startswith("fc_")
    assert evs[-1]["response"]["output"] == [item]


def test_custom_tool_call_comes_back_as_custom_tool_call_item():
    b = ResponseBuilder("m", {"apply_patch": PATCH_TOOL})
    list(b.feed(chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "apply_patch", "arguments": json.dumps({"input": "*** Begin Patch"})}}]})))
    done = [e for e in events(b.finish()) if e["type"] == "response.output_item.done"][0]
    assert done["item"]["type"] == "custom_tool_call"
    assert done["item"]["input"] == "*** Begin Patch" and done["item"]["call_id"] == "c1"


def test_text_then_tool_call_index_output_items_in_order():
    b = ResponseBuilder("m")
    list(b.feed(chunk({"content": "Running ls"})))
    list(b.feed(chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "shell", "arguments": "{}"}}]})))
    done = [e for e in events(b.finish()) if e["type"] == "response.output_item.done"]
    assert [d["output_index"] for d in done] == [0, 1]
    assert [d["item"]["type"] for d in done] == ["message", "function_call"]


def test_blocking_completion_feeds_the_same_builder():
    b = ResponseBuilder("m")
    list(b.feed({"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1}}))
    resp = b.response_object()
    assert resp["status"] == "completed"
    assert resp["output"][0]["type"] == "function_call" and resp["output"][0]["call_id"] == "c1"
    assert resp["usage"]["total_tokens"] == 4


def test_upstream_failure_emits_response_failed_after_partial_text():
    b = ResponseBuilder("m")
    list(b.feed(chunk({"content": "partial"})))
    evs = events(b.finish(error="stream stalled"))
    assert evs[-1]["type"] == "response.failed"
    assert evs[-1]["response"]["status"] == "failed"
    assert evs[-1]["response"]["error"]["message"] == "stream stalled"
    assert "response.completed" not in [e["type"] for e in evs]
