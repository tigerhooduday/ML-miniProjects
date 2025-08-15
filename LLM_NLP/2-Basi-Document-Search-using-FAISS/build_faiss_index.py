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
