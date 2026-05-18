from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Missing sentence-transformers. Install with: pip install -r requirements-local.txt") from exc
    return SentenceTransformer(model_name)


def embed_texts(model, texts: list[str], *, query: bool = False) -> np.ndarray:
    prefix = "query: " if query else "passage: "
    vectors = model.encode(
        [prefix + text for text in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


def build_vector_index(corpus_path: Path, output_dir: Path, model_name: str, batch_size: int) -> dict[str, Any]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Missing faiss-cpu. Install with: pip install -r requirements-local.txt") from exc

    docs = read_jsonl(corpus_path)
    if not docs:
        raise RuntimeError("Corpus is empty. Build output/corpus.jsonl first.")

    model = load_embedding_model(model_name)
    vectors: list[np.ndarray] = []
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        vectors.append(embed_texts(model, [doc["text"] for doc in batch]))
        print(f"vector batch {start // batch_size + 1}: {min(start + batch_size, len(docs))}/{len(docs)}")

    matrix = np.vstack(vectors).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    output_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_dir / "faiss.index"))
    (output_dir / "vector_metadata.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "corpus_path": str(corpus_path),
        "embedding_model": model_name,
        "dimension": int(matrix.shape[1]),
        "doc_count": len(docs),
    }
    (output_dir / "vector_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def vector_search(query: str, vector_dir: Path, top_k: int = 8) -> list[dict[str, Any]]:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Missing faiss-cpu. Install with: pip install -r requirements-local.txt") from exc

    manifest = json.loads((vector_dir / "vector_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((vector_dir / "vector_metadata.json").read_text(encoding="utf-8"))
    index = faiss.read_index(str(vector_dir / "faiss.index"))
    model = load_embedding_model(manifest["embedding_model"])
    query_vector = embed_texts(model, [query], query=True)
    scores, indices = index.search(query_vector, top_k)
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append({**metadata[idx], "score": round(float(score), 6), "search_source": "vector"})
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build/query a local FAISS vector index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--corpus", default="output/corpus.jsonl")
    build_parser.add_argument("--output-dir", default="output/retrieval/vector-local")
    build_parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    build_parser.add_argument("--batch-size", type=int, default=64)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query")
    query_parser.add_argument("--vector-dir", default="output/retrieval/vector-local")
    query_parser.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_vector_index(Path(args.corpus), Path(args.output_dir), args.model, args.batch_size), indent=2))
        return
    print(json.dumps(vector_search(args.query, Path(args.vector_dir), args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
