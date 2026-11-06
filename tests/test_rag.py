import pytest

from coffee_assistant import rag

REWRITE_TOKENS = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
ANSWER_TOKENS = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


@pytest.fixture
def search_calls(monkeypatch):
    # Shared setup: every test that asks for `search_calls` get fresh fakes
    calls = []
    replies = [("espresso crema", REWRITE_TOKENS), ("Crema is the foam.", ANSWER_TOKENS)]
   
    def fake_search(query, num_results=10):
        calls.append(query)
        return []

    def fake_llm(prompt, model="gpt-4o-mini"):
        # 1st call = rewrite, 2nd call = answer, so the test can tell them apart
        return replies.pop(0)

    monkeypatch.setattr(rag, "search", fake_search)
    monkeypatch.setattr(rag, "llm", fake_llm)
    return calls

 

def test_rewrite_goes_to_search_original_goes_to_prompt(search_calls):
    result = rag.rag("expreso crma?")

    assert search_calls == ["espresso crema"]
    assert "expreso crma?" in result["prompt"]


def test_tokens_from_both_llm_calls_are_summed(search_calls):
    result = rag.rag("anything")

    assert result["prompt_tokens"] == 110 
    assert result["completion_tokens"] == 55 
    assert result["total_tokens"] == 165
    