"""GLM tool calls after a ``<tool_call>`` written as text (0145): a mention in the reasoning or the answer is not a
call and does not swallow the real call after it, streamed in any chunk size or parsed whole; a call whose argument
value holds ``<tool_call>`` or ``</think>`` (a file write) is still one call."""

import pytest

from tensorfold.cuda.reply_text import THINK_CALL_HOLD, GlmCallStreamer, ThinkSplit, hide_tool_calls, parse_tool_calls

TOOLS = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": props}}}
         for name, props in [("grep", {"pattern": {"type": "string"}, "path": {"type": "string"}}),
                             ("write", {"content": {"type": "string"}})]]
GREP = ('<tool_call>grep<arg_key>pattern</arg_key><arg_value>json_schema</arg_value>'
        '<arg_key>path</arg_key><arg_value>src</arg_value></tool_call>')
GREP_ARGS = '{"pattern":"json_schema","path":"src"}'
WRITE_VALUE = "line\n" * 400 + "has a <tool_call> and </think> inside"
WRITE = f"<tool_call>write<arg_key>content</arg_key><arg_value>{WRITE_VALUE}</arg_value></tool_call>"


def reply(text: str, *, stream: bool, step: int) -> tuple[str, str, list[tuple[str, str]], str]:
    """(reasoning, content, calls, streamed content) as the CUDA server's ``run`` splits and parses a thinking reply."""

    calls_stream = GlmCallStreamer(TOOLS) if stream else None
    think = ThinkSplit(THINK_CALL_HOLD if stream else None)
    shown = ""
    for end in range(step, len(text) + step, step):
        _, answer = think(text[:end], False)
        visible = hide_tool_calls(answer, finished=False)
        shown = visible if len(visible) > len(shown) else shown
        if calls_stream is not None and think.stream_from is not None:
            calls_stream.feed(text[:end][think.stream_from:])
    reasoning, answer = think(text, True)
    content, calls = parse_tool_calls(answer, TOOLS)
    if calls_stream is not None and think.stream_from is not None:
        calls_stream.feed(text[think.stream_from:])
        calls = [*calls_stream.calls[:calls_stream.sent], *calls_stream.rest(TOOLS)] or None
    return reasoning, content, [(c["function"]["name"], c["function"]["arguments"]) for c in calls or []], shown


CASES = {
    "call after think close": ("Let me look.\n</think>\n" + GREP, "grep"),
    "call right after think close": ("Let me look.</think>" + GREP, "grep"),
    "call inside think block": ("Let me grep.\n" + GREP + "\n</think>", "grep"),
    "mention in reasoning": ("The grammar must allow <tool_call> after reasoning.\n</think>\n" + GREP, "grep"),
    "mention in long reasoning": ("Allow <tool_call> here. " + "x " * 700 + "Check.\n</think>\n" + GREP, "grep"),
    "closed mention in reasoning": ("It parses <tool_call>...</tool_call> blocks.\n</think>\n" + GREP, "grep"),
    "mention in answer": ("Done.\n</think>\nI will emit a <tool_call> now.\n" + GREP, "grep"),
    "write whose value holds the tags": ("Write it.\n</think>\n" + WRITE, "write"),
    "write inside think block": ("Write it.\n" + WRITE, "write"),
}


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("step", [1, 3, 7])
@pytest.mark.parametrize("case", list(CASES))
def test_one_call(case, step, stream):
    text, name = CASES[case]
    _, content, calls, shown = reply(text, stream=stream, step=step)
    assert [c[0] for c in calls] == [name]
    assert "<tool_call>grep" not in content and "</think>" not in content and "</think>" not in shown
    if name == "grep":
        assert calls[0][1] == GREP_ARGS
    else:
        assert calls[0][1].endswith('line\\nhas a <tool_call> and </think> inside"}')


def test_mention_stays_in_the_reasoning():
    reasoning, content, calls, _ = reply(CASES["mention in long reasoning"][0], stream=True, step=7)
    assert reasoning.startswith("Allow <tool_call> here.") and reasoning.rstrip().endswith("Check.")
    assert content == "" and len(calls) == 1


def test_mention_in_answer_is_text():
    _, content, calls, shown = reply(CASES["mention in answer"][0], stream=True, step=1)
    assert content == "I will emit a <tool_call> now." and shown.strip() == content and len(calls) == 1
