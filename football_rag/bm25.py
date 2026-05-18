from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    value = str(text or "").casefold()
    value = unicodedata.normalize("NFD", value)
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize_text(text)) if len(token) > 1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def searchable_text(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    weighted_parts = [
        str(metadata.get("team_name") or metadata.get("team") or ""),
        str(metadata.get("player") or metadata.get("name") or ""),
        str(metadata.get("club") or ""),
        str(doc.get("doc_type") or ""),
        str(doc.get("text") or ""),
    ]
    return "\n".join(part for part in weighted_parts if part)


def build_index(corpus_path: Path) -> dict[str, Any]:
    docs = read_jsonl(corpus_path)
    indexed_docs: list[dict[str, Any]] = []
    document_frequency: Counter[str] = Counter()
    total_length = 0

    for doc in docs:
        tokens = tokenize(searchable_text(doc))
        term_freqs = Counter(tokens)
        total_length += len(tokens)
        for token in term_freqs:
            document_frequency[token] += 1
        indexed_docs.append({**doc, "term_freqs": dict(term_freqs), "doc_len": len(tokens)})

    return {
        "corpus_path": str(corpus_path),
        "doc_count": len(indexed_docs),
        "avg_doc_len": total_length / len(indexed_docs) if indexed_docs else 0.0,
        "document_frequency": dict(document_frequency),
        "documents": indexed_docs,
    }


def save_index(index: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_path: Path) -> dict[str, Any]:
    return json.loads(index_path.read_text(encoding="utf-8"))


def bm25_score(
    query_tokens: list[str],
    term_freqs: dict[str, int],
    doc_len: int,
    avg_doc_len: float,
    doc_count: int,
    document_frequency: dict[str, int],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not query_tokens or not avg_doc_len:
        return 0.0
    score = 0.0
    for token in query_tokens:
        tf = term_freqs.get(token, 0)
        if not tf:
            continue
        df = document_frequency.get(token, 0)
        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
    return score


def query_index(index: dict[str, Any], query: str, top_k: int = 8) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    results: list[dict[str, Any]] = []
    for doc in index["documents"]:
        score = bm25_score(
            query_tokens,
            doc["term_freqs"],
            doc["doc_len"],
            index["avg_doc_len"],
            index["doc_count"],
            index["document_frequency"],
        )
        if score <= 0:
            continue
        clean_doc = {key: value for key, value in doc.items() if key not in {"term_freqs", "doc_len"}}
        results.append({**clean_doc, "score": round(score, 6), "search_source": "bm25"})
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build/query a local BM25 index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--corpus", default="output/corpus.jsonl")
    build_parser.add_argument("--output", default="output/retrieval/bm25_index.json")

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("query")
    query_parser.add_argument("--index", default="output/retrieval/bm25_index.json")
    query_parser.add_argument("--top-k", type=int, default=8)

    args = parser.parse_args()
    if args.command == "build":
        index = build_index(Path(args.corpus))
        save_index(index, Path(args.output))
        print(json.dumps({"documents": index["doc_count"], "output": args.output}, ensure_ascii=False, indent=2))
        return

    print(json.dumps(query_index(load_index(Path(args.index)), args.query, args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
