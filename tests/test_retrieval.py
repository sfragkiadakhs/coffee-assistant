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
