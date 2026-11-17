from types import SimpleNamespace as NS

import pytest

from coffee_assistant import rag

FAKE_RESULTS = {"results": [
    {"title": "Coffee prices", "url": "https://example.com/prices",
     "content": "Arabica costs $3.80/lb."},
    {"title": "Coffee news", "url": "https://example.org/news",
     "content": "Prices rose in November."},
]}


@pytest.fixture
def sent(monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, json=json, timeout=timeout)
        return NS(raise_for_status=lambda: None, json=lambda: FAKE_RESULTS)

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(rag.requests, "post", fake_post)
    return sent


def test_sends_query_and_key_to_tavily(sent):
    rag.web_search("arabica price today")
    assert sent["url"] == "https://api.tavily.com/search"
    assert sent["headers"]["Authorization"] == "Bearer test-key"
    assert sent["json"]["query"] == "arabica price today"
    assert sent["timeout"] is not None


def test_answer_names_each_source(sent):
    text = rag.web_search("arabica price today")
    for result in FAKE_RESULTS["results"]:
        assert result["content"] in text
        assert result["url"] in text
