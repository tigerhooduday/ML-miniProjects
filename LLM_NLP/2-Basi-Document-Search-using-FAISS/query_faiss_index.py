# query_faiss_index.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.pkl"
OUTPUT_FILE = "search_results.md"

def load_index_meta(idx_path=INDEX_PATH, meta_path=META_PATH):
    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["metadatas"]

def search(query, k=5, min_score=None):
    model = SentenceTransformer(MODEL_NAME)
    index, chunks, metas = load_index_meta()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        if min_score is not None and score < min_score:
            continue
        results.append({
            "score": float(score),
            "text": chunks[idx],
            "meta": metas[idx]
        })
    return results

if __name__ == "__main__":
    print("🔍 FAISS Search - Type your query (or 'exit' to quit)")
    
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)  # clear old results

    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("✅ Exiting search.")
            break

        res = search(query, k=5, min_score=None)
        if not res:
            print("No results found.")
            continue

        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n=== Query: {query} ===\n")
            for i, r in enumerate(res, 1):
                header = f"Result {i} | score={r['score']:.4f} | {r['meta']['source']} page:{r['meta']['page']} chunk:{r['meta']['chunk_id']}"
                snippet = r["text"].replace("\n", " ")
                print(header)
                print(snippet[:500] + "...\n")
                f.write(header + "\n" + snippet + "\n\n")

        print(f"💾 Results saved to {OUTPUT_FILE}")
