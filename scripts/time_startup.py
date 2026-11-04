import time

t0 = time.perf_counter()
from coffee_assistant.embedder import Embedder
embedder = Embedder()
t1 = time.perf_counter()


from coffee_assistant import retrieval
t2 = time.perf_counter()

chunks = retrieval.get_index().docs
t3 = time.perf_counter()


retrieval.get_vector_index()
t4 = time.perf_counter()


print(f"model load: {t1 - t0:.1f} s")
print(f"retrieval(lib) load: {t2 - t1:.1f} s")
print(f"read chunks: {t3 - t2:.1f} s")
print(f"embed all: {t4 - t3:.1f} s")
