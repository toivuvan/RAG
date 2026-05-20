import json, os, re, pickle, time
from pathlib import Path
import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
os.makedirs(INDEX_DIR, exist_ok=True)

CHUNK_FILES = [
    ("all_players.jsonl",     "player_stats"),
    ("all_teams.jsonl",       "team_stats"),
    ("all_matches.jsonl",     "match_result"),
    ("transfers_chunks.jsonl","transfer"),
    ("schedule_chunks.jsonl", "schedule"),
]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE  = 128


def normalize_record(raw: dict, source_type: str, global_idx: int) -> dict:
    """Chuẩn hoá 2 schema khác nhau về 1 format duy nhất."""
    # Schema A: all_matches / all_teams / all_players
    if "content" in raw:
        text     = raw["content"]
        chunk_id = raw.get("chunk_id") or raw.get("doc_id") or f"{source_type}_{global_idx:05d}"
        meta     = raw.get("metadata", {})

    # Schema B: transfers_chunks / schedule_chunks
    elif "text" in raw:
        text     = raw["text"]
        chunk_id = raw.get("id") or f"{source_type}_{global_idx:05d}"
        meta     = raw.get("metadata", {})
        # absorb top-level team/source fields into meta
        for k in ("team", "source"):
            if k in raw:
                meta[k] = raw[k]
    else:
        return None

    text = text.strip()
    if not text:
        return None

    return {
        "id":          chunk_id,
        "text":        text,
        "source_type": source_type,
        "metadata":    meta,
        "global_idx":  global_idx,
        "char_count":  len(text),
    }


def load_and_normalize() -> list[dict]:
    """Đọc toàn bộ file chunk JSONL và chuẩn hóa thành một corpus chung."""
    corpus = []
    print("Loading & normalizing chunks:\n")
    for filename, source_type in CHUNK_FILES:
        path = CHUNKS_DIR / filename
        if not path.exists():
            print(f"  [SKIP] {filename} not found")
            continue

        count_ok = count_skip = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rec = normalize_record(raw, source_type, len(corpus))
                if rec:
                    corpus.append(rec)
                    count_ok += 1
                else:
                    count_skip += 1

        print(f"  ✓ {filename:<35} {count_ok:>5} ok  {count_skip:>4} skip")

    return corpus


def build_embeddings(corpus: list[dict]) -> np.ndarray:
    """Tạo vector embedding cho toàn bộ text chunk bằng SentenceTransformer."""
    from sentence_transformers import SentenceTransformer
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in corpus]
    print(f"Embedding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    return embeddings


def build_faiss(embeddings: np.ndarray) -> "faiss.Index":
    """Tạo FAISS index dùng inner product trên vector đã normalize."""
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) 
    index.add(embeddings)
    # print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


STOPWORDS = {
    "the","a","an","in","of","to","for","and","or","is","was",
    "were","are","be","been","at","by","from","on","with","this","that",
    "it","he","she","they","their","his","her","its",
}

def tokenize(text: str) -> list[str]:
    """Tokenize text cho BM25: lowercase, bỏ stopword và token quá ngắn."""
    tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_bm25(corpus: list[dict]):
    """Tạo BM25 sparse index từ corpus đã tokenize."""
    from rank_bm25 import BM25Okapi
    print("Building BM25 index...")
    tokenized = [tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def main():
    """Chạy toàn bộ bước normalize, embedding, FAISS index và BM25 index."""

    # Normalize
    corpus = load_and_normalize()
    print(f"\nTotal corpus: {len(corpus)} chunks")
    if not corpus:
        raise FileNotFoundError(
            f"No chunks found in {CHUNKS_DIR}. Run chunker first: python src/chunker.py"
        )

    # Lưu corpus đã chuẩn hoá
    corpus_path = CHUNKS_DIR / "corpus_unified.jsonl"
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved: {corpus_path}")

    # Embedding
    embeddings = build_embeddings(corpus)

    # FAISS
    faiss_index = build_faiss(embeddings)
    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss_index.bin"))

    # ID map để tra cứu metadata khi biết id của chunk
    id_map = {
        c["global_idx"]: {
            "id":          c["id"],
            "text":        c["text"],
            "source_type": c["source_type"],
            "metadata":    c["metadata"],
        }
        for c in corpus
    }
    with open(INDEX_DIR / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)

    # BM25 sparse index để truy vấn theo keyword (không dùng embedding)
    bm25, tokenized_corpus = build_bm25(corpus)
    with open(INDEX_DIR / "bm25_corpus.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized_corpus}, f)

    print(f"""
    INDEXING COMPLETE ✓
    corpus_unified.jsonl  {len(corpus)} chunks
    faiss_index.bin       {len(corpus)} vectors (dim=384)
    bm25_corpus.pkl       {len(corpus)} documents
    id_map.json           {len(corpus)} entries
""")

if __name__ == "__main__":
    main()
