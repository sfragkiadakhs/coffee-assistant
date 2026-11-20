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


def test_long_result_is_cut_but_keeps_its_source(monkeypatch):
    long_page = "menu " * 1000  # 5,000 characters, like a page full of menus
    fake = {"results": [{"title": "Shop", "url": "https://example.com/shop",
                         "content": long_page}]}
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(rag.requests, "post",
                        lambda *a, **kw: NS(raise_for_status=lambda: None, json=lambda: fake))

    text = rag.web_search("arabica price today")

    assert long_page[:rag.MAX_RESULT_CHARS] in text
    assert long_page[:rag.MAX_RESULT_CHARS + 1] not in text
    assert "https://example.com/shop" in text
