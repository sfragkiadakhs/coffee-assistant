import numpy as np

from coffee_assistant import retrieval
from coffee_assistant.ingest import DATA_DIR


# The slow way, once: read chunks + embed all of them
embedder, chunks, X = retrieval.get_vector_index()
chunk_ids = [chunk["chunk_id"] for chunk in chunks]


np.savez(
    DATA_DIR / "embeddings.npz",
    X = X.astype(np.float32),       # float32 = half the size of float64, same search result
    chunk_ids = np.array(chunk_ids),
)
print(f"saved {X.shape[0]} vectors of size {X.shape[1]}")