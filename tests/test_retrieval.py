from coffee_assistant.retrieval import rrf


def test_rrf_doc_in_both_lists_ranks_first():
    # Arrange: two ranked lists, like keyword + vector search return
    keyword = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    vector = [{"chunk_id": "b"}, {"chunk_id": "c"}]

    # Act: fuse them
    result = rrf([keyword, vector])

    # Assert: b is in both lists -> 1.5 points, a = 1.0, c = 0.5
    ids = [doc["chunk_id"] for doc in result]
    assert ids == ["b", "a", "c"]


def test_rrf_num_results_keeps_top_n():
    # Arrange: two ranked lists, like keyword + vector search return
    keyword = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    vector = [{"chunk_id": "b"}, {"chunk_id": "c"}]

    # Act: fuse them
    result = rrf([keyword, vector], num_results=2)

    # Assert: only the top 2 stay (b 1.5, a 1.0); c drops out
    ids = [doc["chunk_id"] for doc in result]
    assert ids == ["b", "a"]


import numpy as np

from coffee_assistant import retrieval


class FakeEmbedder:
    # Always turns the question into the same vector: "about espresso"
    def encode(self, text):
        return np.array([1.0, 0.0])


def test_vector_search_ranks_closest_chunk_first(monkeypatch):
    # Arrange 3 chunks with hand-made vectors
    chunks = [{"chunk_id": "milk"}, {"chunk_id": "espresso"}, {"chunk_id": "grinder"}]
    X = np.array([
        [0.0, 1.0], # milk: score 0.0
        [0.9, 0.1], # espresso: score 0.9
        [0.5, 0.5], # grinder: score 0.5
    ])
    # Swap the slow real index for our fake one, only during the test
    monkeypatch.setattr(retrieval, "get_vector_index", lambda: (FakeEmbedder(), chunks, X))

    # Act
    result = retrieval.vector_search("anything", num_results=2)

    # Assert: highest score first, only 2 results
    ids = [doc["chunk_id"] for doc in result]
    assert ids == ["espresso", "grinder"]