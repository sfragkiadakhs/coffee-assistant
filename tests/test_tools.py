import json
from types import SimpleNamespace as NS

import pytest

from coffee_assistant import rag


def tool_call_reply(call_id, query):
    calls = NS(id=call_id, type="function",
              function=NS(name="web_search", arguments=json.dumps({"query":query})))
    return NS(content=None, tool_calls=[calls])


def text_reply(text):
    return NS(content=text, tool_calls=None)


@pytest.fixture
def seen(monkeypatch):
    seen = {"searches": [], "messages": []}
    replies = [tool_call_reply("call_1", "arabica price today"),
               text_reply("About $3.80/lb (web: example.com)")]

    def fake_chat(messages, model="gpt-4o-mini"):
        seen["messages"].append(list(messages))
        return replies.pop(0)

    def fake_web_search(query):
        seen["searches"].append(query)
        return "Arabica $3.80/lb, source: example.com"

    monkeypatch.setattr(rag, "chat", fake_chat)
    monkeypatch.setattr(rag, "web_search", fake_web_search)
    return seen


def test_tool_runs_with_the_models_arguments(seen):
    rag.answer_with_tools("What does arabica cost today?")

    assert seen["searches"] == ["arabica price today"]


def test_tool_result_goes_back_with_the_same_id(seen):
    answer = rag.answer_with_tools("What does arabica cost today?")

    last = seen["messages"][1][-1]
    assert last["role"] == "tool"
    assert last["tool_call_id"] == "call_1"
    assert "3.80" in last["content"]
    assert answer == "About $3.80/lb (web: example.com)"