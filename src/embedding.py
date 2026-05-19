"""
STEP 2 — NORMALIZE + EMBED + INDEX
====================================
Input : 5 JSONL files từ bước chunking
Output:
  data/chunks/corpus_unified.jsonl   ← tất cả chunks với schema thống nhất
  data/index/faiss_index.bin         ← FAISS dense index
  data/index/id_map.json             ← chunk_id → text/metadata
  data/index/bm25_corpus.pkl         ← BM25 tokenized corpus
"""

import json, os, re, pickle, time
from pathlib import Path
import numpy as np

CHUNKS_DIR  = "./data/chunks"
INDEX_DIR   = "./data/index"
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


# ─────────────────────────────────────────
#  NORMALIZE — thống nhất về {id, text, metadata, source_type}
# ─────────────────────────────────────────

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
    corpus = []
    print("Loading & normalizing chunks:\n")
    for filename, source_type in CHUNK_FILES:
        path = os.path.join(CHUNKS_DIR, filename)
        if not os.path.exists(path):
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


# ─────────────────────────────────────────
#  EMBED — sentence-transformers MiniLM
# ─────────────────────────────────────────

def build_embeddings(corpus: list[dict]) -> np.ndarray:
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
        normalize_embeddings=True,   # cosine = dot product after L2 norm
    ).astype("float32")

    print(f"Done in {time.time()-t0:.1f}s  shape={embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────
#  FAISS INDEX
# ─────────────────────────────────────────

def build_faiss(embeddings: np.ndarray) -> "faiss.Index":
    import faiss
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)     # Inner Product = cosine (vectors are L2-normalised)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ─────────────────────────────────────────
#  BM25 INDEX
# ─────────────────────────────────────────

STOPWORDS = {
    "the","a","an","in","of","to","for","and","or","is","was",
    "were","are","be","been","at","by","from","on","with","this","that",
    "it","he","she","they","their","his","her","its",
}

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def build_bm25(corpus: list[dict]):
    from rank_bm25 import BM25Okapi
    print("Building BM25 index...")
    tokenized = [tokenize(c["text"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)
    print(f"BM25 index built: {len(tokenized)} documents")
    return bm25, tokenized


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 58)
    print("  STEP 2: NORMALIZE + EMBED + INDEX")
    print("=" * 58)

    # ── 1. Normalize
    corpus = load_and_normalize()
    print(f"\nTotal corpus: {len(corpus)} chunks")

    # Save unified corpus
    corpus_path = os.path.join(CHUNKS_DIR, "corpus_unified.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved: {corpus_path}")

    # ── 2. Embed
    embeddings = build_embeddings(corpus)

    # ── 3. FAISS
    faiss_index = build_faiss(embeddings)
    import faiss
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, "faiss_index.bin"))
    print(f"Saved: {INDEX_DIR}/faiss_index.bin")

    # ── 4. ID map (global_idx → id + text + metadata) for lookup after retrieval
    id_map = {
        c["global_idx"]: {
            "id":          c["id"],
            "text":        c["text"],
            "source_type": c["source_type"],
            "metadata":    c["metadata"],
        }
        for c in corpus
    }
    with open(os.path.join(INDEX_DIR, "id_map.json"), "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False)
    print(f"Saved: {INDEX_DIR}/id_map.json")

    # ── 5. BM25
    bm25, tokenized_corpus = build_bm25(corpus)
    with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized_corpus}, f)
    print(f"Saved: {INDEX_DIR}/bm25_corpus.pkl")

    # ── Summary by source type
    by_src = {}
    for c in corpus:
        by_src.setdefault(c["source_type"], 0)
        by_src[c["source_type"]] += 1

    print(f"\n{'─'*45}")
    print(f"  {'Source type':<25} {'Chunks':>8}")
    print(f"{'─'*45}")
    for src, cnt in sorted(by_src.items(), key=lambda x: -x[1]):
        print(f"  {src:<25} {cnt:>8}")
    print(f"{'─'*45}")
    print(f"  {'TOTAL':<25} {len(corpus):>8}")
    print(f"""
{'='*58}
  INDEXING COMPLETE ✓
  corpus_unified.jsonl  {len(corpus)} chunks
  faiss_index.bin       {len(corpus)} vectors (dim=384)
  bm25_corpus.pkl       {len(corpus)} documents
  id_map.json           {len(corpus)} entries
{'='*58}
→ Next: python src/03_generate_qa.py
""")

if __name__ == "__main__":
    main()
