Nice — I’ll walk through **every line** of both files so you know exactly what each part does. I’ll keep the explanations short and practical, grouped by logical blocks. If anything looks confusing, tell me which line and I’ll expand.

# build\_faiss\_index.py

```python
# build_faiss_index.py
import os, glob, pickle, argparse
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm

# Config — tune these
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # overlap between chunks
BATCH_SIZE = 64        # embedding batch size
OUT_INDEX = "data/faiss.index"
OUT_META = "data/meta.pkl"

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    pages = []
    for pnum, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.replace("\r", " ")
        pages.append((pnum+1, text))
    return pages

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= L:
            break
        start = max(end - overlap, start + 1)
    return chunks

def build_index(pdf_folder, out_index=OUT_INDEX, out_meta=OUT_META):
    model = SentenceTransformer(MODEL_NAME)
    all_chunks = []
    metadatas = []  # list of dicts: {source, page, chunk_id}
    files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    print(f"Found {len(files)} pdf files.")

    for path in files:
        pages = extract_text_from_pdf(path)
        for page_num, text in pages:
            chunks = chunk_text(text)
            for i, c in enumerate(chunks):
                all_chunks.append(c)
                metadatas.append({"source": os.path.basename(path), "page": page_num, "chunk_id": i})

    if not all_chunks:
        raise SystemExit("No text extracted from PDFs. Check your PDF files or use OCR for scanned PDFs.")

    # embeddings (float32)
    print("Computing embeddings...")
    embeddings = model.encode(all_chunks, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # normalize -> use inner product for cosine sim
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # exact search, inner product (works as cosine if normalized)
    index.add(embeddings)
    os.makedirs(os.path.dirname(out_index), exist_ok=True)
    faiss.write_index(index, out_index)

    # save metadata & chunks
    with open(out_meta, "wb") as f:
        pickle.dump({"chunks": all_chunks, "metadatas": metadatas}, f)

    print(f"Index saved to {out_index}, metadata saved to {out_meta}. Total chunks: {len(all_chunks)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf_folder", default="pdfs")
    p.add_argument("--out_index", default=OUT_INDEX)
    p.add_argument("--out_meta", default=OUT_META)
    args = p.parse_args()
    build_index(args.pdf_folder, args.out_index, args.out_meta)
```

### Line-by-line explanation (build\_faiss\_index.py)

**Imports & config**

* `import os, glob, pickle, argparse`

  * `os`: filesystem helpers (paths, mkdir).
  * `glob`: find files by pattern (e.g., `*.pdf`).
  * `pickle`: save/load Python objects (metadata & chunks).
  * `argparse`: parse command-line arguments.

* `from PyPDF2 import PdfReader`

  * `PdfReader` reads PDF and extracts text from pages.

* `from sentence_transformers import SentenceTransformer`

  * Loads embedding models (turns text → vectors).

* `import numpy as np`

  * NumPy (imported but not actively used in this script — safe to remove if you want).

* `import faiss`

  * FAISS library: fast vector indexing and nearest-neighbor search.

* `from tqdm import tqdm`

  * Progress bar helper (not required here because `model.encode(..., show_progress_bar=True)` uses its own progress, but harmless).

**Config constants**

* `MODEL_NAME = "all-MiniLM-L6-v2"`

  * Embedding model name (fast & small). Use same model for indexing & querying.

* `CHUNK_SIZE = 1000`

  * Approx characters per chunk. Tune by document style.

* `CHUNK_OVERLAP = 200`

  * How many characters overlap between consecutive chunks (keeps context).

* `BATCH_SIZE = 64`

  * Number of texts passed to `model.encode` at once (affects speed & memory).

* `OUT_INDEX = "data/faiss.index"` and `OUT_META = "data/meta.pkl"`

  * Where the FAISS index and metadata will be saved.

**Function: `extract_text_from_pdf(path)`**

* `reader = PdfReader(path)`

  * Open the PDF file.

* `pages = []`

  * Prepare list to hold `(page_number, text)` tuples.

* `for pnum, page in enumerate(reader.pages):`

  * Iterate through pages; `pnum` starts at 0.

* `text = page.extract_text() or ""`

  * Extract text from the page. If extraction returns `None`, use empty string.

* `text = text.replace("\r", " ")`

  * Replace carriage returns that may break chunking.

* `pages.append((pnum+1, text))`

  * Save page number (1-indexed) and its text.

* `return pages`

  * Returns list of `(page_num, text)` for the PDF.

**Function: `chunk_text(text, size=..., overlap=...)`**

* `text = text.strip()`

  * Trim leading/trailing whitespace.

* `if not text: return []`

  * If there's no text, return empty list (no chunks).

* `chunks = []` and `start = 0` and `L = len(text)`

  * Prepare variables.

* `while start < L:`

  * Continue until we've processed the whole text.

* `end = start + size`

  * Candidate end index for the chunk.

* `chunk = text[start:end].strip()`

  * Extract the substring and trim it.

* `if chunk: chunks.append(chunk)`

  * Append non-empty chunk.

* `if end >= L: break`

  * If we've reached the end of text, stop the loop.

* `start = max(end - overlap, start + 1)`

  * Move window forward so there is `overlap` characters between consecutive chunks; `start+1` ensures progress if `overlap` is too large.

* `return chunks`

  * Return list of text chunks for that text block.

**Function: `build_index(pdf_folder, out_index=..., out_meta=...)`**

* `model = SentenceTransformer(MODEL_NAME)`

  * Load the embedding model into memory.

* `all_chunks = []` and `metadatas = []`

  * `all_chunks` stores chunk texts (parallel to embeddings).
  * `metadatas` stores dicts with `source` (filename), `page`, `chunk_id`.

* `files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))`

  * Find all `*.pdf` files inside `pdf_folder`, sort them.

* `print(f"Found {len(files)} pdf files.")`

  * Informational print.

* `for path in files:`

  * Loop through each PDF file path.

* `pages = extract_text_from_pdf(path)`

  * Get page texts for this PDF.

* `for page_num, text in pages:`

  * Loop each page returned.

* `chunks = chunk_text(text)`

  * Break page text into chunks.

* `for i, c in enumerate(chunks):`

  * For each chunk, record text and metadata.

* `all_chunks.append(c)`

  * Append chunk text.

* `metadatas.append({"source": os.path.basename(path), "page": page_num, "chunk_id": i})`

  * Append metadata dict (filename, page number, chunk index).

* `if not all_chunks: raise SystemExit("No text extracted from PDFs. Check your PDF files or use OCR for scanned PDFs.")`

  * If nothing was extracted (often scanned PDFs), exit with message.

**Embeddings & FAISS index creation**

* `print("Computing embeddings...")`

  * Informational.

* `embeddings = model.encode(all_chunks, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True)`

  * Convert all chunk texts into embeddings (vectors). Returns NumPy array because `convert_to_numpy=True`.

* `embeddings = embeddings.astype("float32")`

  * FAISS expects `float32`. Ensure correct dtype.

* `faiss.normalize_L2(embeddings)`

  * Normalize each vector to unit length (L2 norm = 1). After this, inner-product search equals cosine similarity.

* `d = embeddings.shape[1]`

  * Dimensionality of embeddings (e.g., 384).

* `index = faiss.IndexFlatIP(d)`

  * Create an *exact* index using inner-product (IP). Since we normalized embeddings, IP ≈ cosine similarity.

* `index.add(embeddings)`

  * Add all vectors into the FAISS index.

* `os.makedirs(os.path.dirname(out_index), exist_ok=True)`

  * Create the directory for the index file (`data/`) if it doesn't exist. `exist_ok=True` avoids error if it already exists.

* `faiss.write_index(index, out_index)`

  * Save the FAISS index binary to disk.

**Save metadata**

* `with open(out_meta, "wb") as f: pickle.dump({"chunks": all_chunks, "metadatas": metadatas}, f)`

  * Save chunk texts and metadata to a pickle file so you can map index ids → original text and sources later.

* `print(f"Index saved to {out_index}, metadata saved to {out_meta}. Total chunks: {len(all_chunks)}")`

  * Print final status.

**Command-line entrypoint**

* `if __name__ == "__main__":`

  * Standard Python entrypoint — code below runs only if the script is executed directly.

* `p = argparse.ArgumentParser()`

  * Create an argument parser.

* `p.add_argument("--pdf_folder", default="pdfs")`

  * Optional CLI argument to specify PDF directory. Defaults to `pdfs`.

* `p.add_argument("--out_index", default=OUT_INDEX)` and `p.add_argument("--out_meta", default=OUT_META)`

  * Optional CLI args to override output paths.

* `args = p.parse_args()`

  * Parse CLI args.

* `build_index(args.pdf_folder, args.out_index, args.out_meta)`

  * Call the main function with parsed arguments.

---

# query\_faiss\_index.py (interactive version)

```python
# query_faiss_index.py
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss.index"
META_PATH = "data/meta.pkl"
OUTPUT_FILE = "search_results.txt"

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
```

### Line-by-line explanation (query\_faiss\_index.py)

**Imports & constants**

* `import faiss` — FAISS library to load index and search.
* `import pickle` — load saved metadata & chunks.
* `from sentence_transformers import SentenceTransformer` — same embedding model used for queries.
* `import os` — filesystem helpers.
* `MODEL_NAME = "all-MiniLM-L6-v2"` — embedding model (must match the one used for indexing).
* `INDEX_PATH = "data/faiss.index"` / `META_PATH = "data/meta.pkl"` — paths produced by the build script.
* `OUTPUT_FILE = "search_results.txt"` — where interactive results are appended for later reading.

**Function: `load_index_meta(idx_path=..., meta_path=...)`**

* `index = faiss.read_index(idx_path)`

  * Load the FAISS index binary into memory.

* `with open(meta_path, "rb") as f: data = pickle.load(f)`

  * Load the metadata & chunks pickle saved earlier.

* `return index, data["chunks"], data["metadatas"]`

  * Return the index, the list of chunk texts, and the list of metadata dicts.

**Function: `search(query, k=5, min_score=None)`**

* `model = SentenceTransformer(MODEL_NAME)`

  * Load the embedding model (notice: in this version model is re-loaded on every call; see optimization note below).

* `index, chunks, metas = load_index_meta()`

  * Load index and metadata (also re-loaded on every call in current code).

* `q_emb = model.encode([query], convert_to_numpy=True).astype("float32")`

  * Convert the single query into a 1×D embedding array.

* `faiss.normalize_L2(q_emb)`

  * Normalize the query vector to unit length — necessary because the index was built with normalized vectors.

* `D, I = index.search(q_emb, k)`

  * Search the index for `k` nearest neighbors.

    * `D`: distances/scores (shape 1×k). Since we used inner-product on normalized vectors, `D` ≈ cosine similarities in \[-1, 1].
    * `I`: indices of found vectors (shape 1×k).

* `results = []` — prepare result list.

* `for score, idx in zip(D[0], I[0]):`

  * Iterate over returned scores and indices for the single query.

* `if idx < 0: continue`

  * FAISS may return -1 for missing neighbors; skip them.

* `if min_score is not None and score < min_score: continue`

  * If user provided a minimum score threshold, skip results below it.

* `results.append({...})`

  * Append a dict with `score` (float), `text` (chunk), and `meta` (source/page/chunk\_id).

* `return results`

  * Return the list of matching chunks.

**Interactive CLI**

* `if __name__ == "__main__":` — run below code when script executed.

* `print("🔍 FAISS Search - Type your query (or 'exit' to quit)")` — friendly header.

* `if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)`

  * Remove previous `search_results.txt` so each run starts fresh.

* `while True:` — interactive loop.

* `query = input("\nEnter query: ").strip()`

  * Prompt user, read input, trim whitespace.

* `if query.lower() in ["exit", "quit"]:`

  * If user types `exit` or `quit`, break loop and finish.

* `res = search(query, k=5, min_score=None)`

  * Call search for top 5 results (no minimum score).

* `if not res: print("No results found."); continue`

  * If no results, notify and go back to prompt.

* `with open(OUTPUT_FILE, "a", encoding="utf-8") as f:`

  * Open results file in append mode.

* `f.write(f"\n=== Query: {query} ===\n")`

  * Write a header for this query into the file.

* `for i, r in enumerate(res, 1):`

  * For each returned result:

* `header = f"Result {i} | score={r['score']:.4f} | {r['meta']['source']} page:{r['meta']['page']} chunk:{r['meta']['chunk_id']}"`

  * Build a readable header with score and source info.

* `snippet = r["text"].replace("\n", " ")`

  * Replace newlines in chunk text so file/console output is neat.

* `print(header)` and `print(snippet[:500] + "...\n")`

  * Print header and first 500 chars of snippet to terminal for quick reading.

* `f.write(header + "\n" + snippet + "\n\n")`

  * Save header + full snippet into the output file.

* `print(f"💾 Results saved to {OUTPUT_FILE}")`

  * Inform the user that results are saved.

---

## Important notes & small optimizations

1. **Model / index loading overhead**

   * Current `search()` **loads the SentenceTransformer and reads the FAISS index on every query**. That’s slow. Usually you should load the model and index **once** at program start, e.g.:

```python
# load once (put at top-level inside __main__)
model = SentenceTransformer(MODEL_NAME)
index, chunks, metas = load_index_meta()
# then search() uses those globals instead of reloading
```

2. **Scores meaning**

   * Because we normalized vectors and used inner-product (`IndexFlatIP`), returned `score` ≈ cosine similarity ∈ \[-1, 1]. Higher is better. In practice:

     * > 0.65 strong match, 0.45–0.65 moderate, <0.3 weak/noisy (model-dependent).

3. **Large corpora**

   * `IndexFlatIP` is exact but not memory/CPU efficient at large scale. For many chunks (100k+) use FAISS IVF/HNSW for speed.

4. **Scanned PDFs (OCR)**

   * If `extract_text_from_pdf` returns empty pages, your PDF is likely scanned images → use OCR (`pytesseract`) before indexing.

5. **Pickle safety**

   * Pickle is convenient but not secure for untrusted files. Only load meta.pkl you created yourself.

6. **Removing unused imports**

   * `numpy as np` and `tqdm` are not required in the build script as written; remove if you want a cleaner file.

---

If you want, I can:

* Patch the interactive query script to load model/index once (faster interactive experience) and paste the updated file, or
* Add a tiny function to **highlight query words** in the saved `search_results.txt`, or
* Show how to run a few example queries and interpret the numeric scores.

Which one next?
