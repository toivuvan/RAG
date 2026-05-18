from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = [
    "data/processed/all_documents.jsonl",
    "data/processed/chunks_teams.jsonl",
    "data/processed/chunks_players.jsonl",
    "data/processed/chunks_goalkeepers.jsonl",
    "data/processed/chunks_matches.jsonl",
    "data/raw/epl_transfers/epl_transfers/corpus.jsonl",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def document_text(record: dict[str, Any]) -> str:
    if record.get("content"):
        return str(record["content"]).strip()
    if record.get("text"):
        return str(record["text"]).strip()

    parts: list[str] = []
    for key in ("name", "team", "position", "nationality", "season", "narrative"):
        if record.get(key) is not None:
            parts.append(f"{key}: {record[key]}")
    if record.get("stats"):
        parts.append(f"stats: {compact_json(record['stats'])}")
    if record.get("metadata"):
        parts.append(f"metadata: {compact_json(record['metadata'])}")
    return "\n".join(parts).strip()


def document_id(record: dict[str, Any], source_file: Path, index: int) -> str:
    for key in ("chunk_id", "id", "doc_id"):
        if record.get(key):
            return str(record[key])
    return f"{source_file.stem}_{index}"


def normalize_record(record: dict[str, Any], source_file: Path, index: int) -> dict[str, Any] | None:
    text = document_text(record)
    if not text:
        return None

    metadata = dict(record.get("metadata") or {})
    for key in (
        "type",
        "doc_type",
        "chunk_type",
        "name",
        "team",
        "position",
        "season",
        "source",
        "source_file",
        "club",
        "player",
    ):
        if record.get(key) is not None and key not in metadata:
            metadata[key] = record[key]

    return {
        "chunk_id": document_id(record, source_file, index),
        "doc_id": str(record.get("doc_id") or record.get("id") or document_id(record, source_file, index)),
        "doc_type": str(record.get("doc_type") or record.get("type") or metadata.get("doc_type") or "document"),
        "text": text,
        "source_file": str(source_file.as_posix()),
        "metadata": metadata,
    }


def build_corpus(source_paths: list[Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source_path in source_paths:
        for index, record in enumerate(read_jsonl(source_path)):
            normalized = normalize_record(record, source_path, index)
            if not normalized:
                continue
            dedupe_key = (normalized["chunk_id"], normalized["text"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            documents.append(normalized)
    return documents


def save_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build a normalized football RAG corpus.")
    parser.add_argument("--sources", nargs="*", default=DEFAULT_SOURCES)
    parser.add_argument("--output", default="output/corpus.jsonl")
    args = parser.parse_args()

    documents = build_corpus([Path(source) for source in args.sources])
    save_jsonl(documents, Path(args.output))
    print(json.dumps({"documents": len(documents), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
