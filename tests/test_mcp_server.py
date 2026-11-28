import pytest
from mcp import Client

import mcp_server
from coffee_assistant import rag


@pytest.fixture
def fake_rag(monkeypatch):
    seen = []

    def fake(question):
        seen.append(question)
        return {"answer": "Espresso has about 63 mg of caffeine (source: coffee database)",
                "total_tokens": 120}

    monkeypatch.setattr(rag, "rag", fake)
    return seen


@pytest.mark.anyio
async def test_tool_is_listed_with_description_and_schema():
    async with Client(mcp_server.mcp, raise_exceptions=True) as client:
        result = await client.list_tools()

    tool = result.tools[0]
    assert [t.name for t in result.tools] == ["ask_coffee_assistant"]
    assert tool.description  # the docstring is what the AI app reads to decide when to call it
    assert tool.input_schema["properties"]["question"]["type"] == "string"
    assert tool.input_schema["required"] == ["question"]


@pytest.mark.anyio
async def test_tool_returns_only_the_answer_text(fake_rag):
    async with Client(mcp_server.mcp, raise_exceptions=True) as client:
        result = await client.call_tool("ask_coffee_assistant",
                                        {"question": "How much caffeine is in espresso?"})

    assert fake_rag == ["How much caffeine is in espresso?"]
    assert result.content[0].text == "Espresso has about 63 mg of caffeine (source: coffee database)"
