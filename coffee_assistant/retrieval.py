import numpy as np

from coffee_assistant import ingest
from coffee_assistant.embedder import Embedder


_index = None
_embedder = None
_chunks = None
_X = None


def get_index():
    global _index
    if _index is None:
        _index = ingest.load_index()
    return _index


def get_vector_index():
    """Lazily build the chunk embedding matrix, rebuilt fresh each process run."""
    global _embedder, _chunks, _X
    if _X is None:
        _embedder = Embedder()
        _chunks = get_index().docs
        texts = [chunk['content'] for chunk in _chunks]

        vectors = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vectors.extend(_embedder.encode_batch(batch))
        _X = np.array(vectors)

    return _embedder, _chunks, _X


def keyword_search(query, num_results=10):
    return get_index().search(
        query=query,
        filter_dict={},
        boost_dict={},
        num_results=num_results,
    )


def vector_search(query, num_results=10):
    embedder, chunks, X = get_vector_index()
    v_query = embedder.encode(query)
    scores = X.dot(v_query)
    idx = np.argsort(-scores)[:num_results]
    return [chunks[i] for i in idx]


def rrf(result_lists, k=1, num_results=10):
    """Reciprocal Rank Fusion. k=1 (not the literature default of 60) because our
    per-method result lists only have ~10 candidates; see retrieval-eval.ipynb for
    why a larger k flattens the ranking signal at this scale."""
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = doc["chunk_id"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=1, num_results=10):
    # Each method always fetches 10 candidates for RRF to fuse, regardless of the
    # final num_results requested — a bigger candidate pool makes for a better fusion.
    text_results = keyword_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k, num_results=num_results)
