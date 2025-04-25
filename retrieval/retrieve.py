import os
import json
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------- Config ---------------
TEST_DIR    = "test"
INDEX_PATH  = "model/embeddings_index.faiss"
META_PATH   = "model/chunks.json"
TOP_K       = 5

# 1. Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    file_paths = json.load(f)

# 1a. Normalize paths (remove leading "../" so they point under project root)
file_paths = [
    p[3:] if p.startswith("../") else p
    for p in file_paths
]
# 2. Read documents
documents = []
for path in file_paths:
    with open(path, "r", encoding="utf-8", errors="ignore") as fd:
        documents.append(fd.read())

# 3. Re‐fit TF-IDF on all docs
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(documents).toarray().astype("float32")

# 4. Load FAISS index
index = faiss.read_index(INDEX_PATH)

# 5. Define a helper to embed a query
def embed_query(query: str):
    vec = vectorizer.transform([query]).toarray().astype("float32")
    return vec

# 6. Run a search
def retrieve(query: str, top_k=TOP_K):
    q_vec = embed_query(query)
    D, I = index.search(q_vec, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        path = file_paths[idx]
        snippet = "\n".join(open(path).read().splitlines()[:10])  # first 10 lines
        results.append((dist, path, snippet))
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrieve similar test files")
    parser.add_argument("query", help="Natural-language scenario (e.g. login)")
    parser.add_argument("--k", type=int, default=TOP_K, help="Number of results")
    args = parser.parse_args()

    for dist, path, snippet in retrieve(args.query, args.k):
        print(f"\n— match (L2 distance {dist:.4f}): {path}\n")
        print(snippet)
        print("…")