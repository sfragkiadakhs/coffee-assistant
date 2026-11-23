import pytest

from coffee_assistant import rag

REWRITE_TOKENS = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
ANSWER_TOKENS = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


@pytest.fixture
def seen(monkeypatch):
    # Shared setup: every test that asks for `seen` gets fresh fakes
    seen = {"searches": [], "answer_prompts": []}

    def fake_search(query, num_results=10):
        seen["searches"].append(query)
        return []

    def fake_llm(prompt, model="gpt-4o-mini"):
        # only the rewrite still uses llm()
        return "espresso crema", REWRITE_TOKENS

    def fake_answer_with_tools(question, model="gpt-4o-mini"):
        # the answer now comes from the tool loop, which can search the web
        seen["answer_prompts"].append(question)
        return "Crema is the foam. (source: coffee database)", ANSWER_TOKENS

    monkeypatch.setattr(rag, "search", fake_search)
    monkeypatch.setattr(rag, "llm", fake_llm)
    monkeypatch.setattr(rag, "answer_with_tools", fake_answer_with_tools)
    return seen


def test_rewrite_goes_to_search_original_goes_to_prompt(seen):
    result = rag.rag("expreso crma?")

    assert seen["searches"] == ["espresso crema"]
    assert "expreso crma?" in result["prompt"]


def test_tokens_from_rewrite_and_answer_are_summed(seen):
    result = rag.rag("anything")

    assert result["prompt_tokens"] == 110
    assert result["completion_tokens"] == 55
    assert result["total_tokens"] == 165


def test_answer_comes_from_the_tool_loop(seen):
    result = rag.rag("What is crema?")

    assert seen["answer_prompts"] == [result["prompt"]]   # the full prompt went to the loop
    assert result["answer"] == "Crema is the foam. (source: coffee database)"


def test_prompt_allows_the_web_and_asks_for_the_source(seen):
    prompt = rag.rag("What does arabica cost today?")["prompt"]

    assert "Use only the facts" not in prompt   # the old rule blocked the web
    assert "source" in prompt.lower()           # the model must say where the answer came from


def test_prompt_keeps_the_bot_on_coffee_and_in_the_users_language(seen):
    # The 22.11 log: off-topic questions (Harvard, linear regression) went to the web,
    # Greek questions got English answers. Only a real run proves the model follows this.
    prompt = rag.rag("Please find me a master's degree in Ulm")["prompt"]

    assert "only answer questions about coffee" in prompt
    assert "isn't about coffee, don't search the web" in prompt
    assert "language of the QUESTION" in prompt
