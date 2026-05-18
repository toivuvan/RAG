from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .retrieve import hybrid_search


SYSTEM_PROMPT = """You are a local RAG assistant for Premier League 2023-24 data.
Answer only from the retrieved context. If the context is not enough, say what is missing.
Keep the answer concise and cite sources as [S1], [S2], ..."""


def build_context(results: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        label_parts = [item.get("doc_type", "document"), item.get("chunk_id", "")]
        team = metadata.get("team_name") or metadata.get("team") or item.get("team")
        if team:
            label_parts.append(str(team))
        blocks.append(
            f"[S{index}] {' | '.join(str(part) for part in label_parts if part)}\n"
            f"Source file: {item.get('source_file')}\n"
            f"Text: {item.get('text', '')}"
        )
    return "\n\n".join(blocks)


def call_local_llm(question: str, context: str, *, base_url: str, model: str, timeout: int) -> str:
    import requests

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nRetrieved context:\n{context}"},
        ],
    }
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer local"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"]).strip()


def extractive_answer(question: str, results: list[dict[str, Any]], *, llm_error: str | None = None) -> str:
    if not results:
        return "Không tìm thấy context phù hợp trong corpus local."
    heading = "Trả lời extractive từ các đoạn retrieve được:"
    if llm_error:
        heading = "Không gọi được local LLM, nên trả lời extractive từ các đoạn retrieve được:"
    lines = [heading]
    for index, item in enumerate(results[:4], start=1):
        text = " ".join(str(item.get("text", "")).split())
        snippet = text[:600] + ("..." if len(text) > 600 else "")
        lines.append(f"[S{index}] {snippet}")
    return "\n\n".join(lines)


def answer_question(
    question: str,
    *,
    bm25_index_path: Path,
    vector_dir: Path,
    mode: str,
    top_k: int,
    use_llm: bool,
    llm_base_url: str,
    llm_model: str,
) -> dict[str, Any]:
    results = hybrid_search(
        question,
        bm25_index_path=bm25_index_path,
        vector_dir=vector_dir,
        mode=mode,
        top_k=top_k,
    )
    context = build_context(results)
    answer = ""
    llm_error = None
    if use_llm:
        try:
            answer = call_local_llm(question, context, base_url=llm_base_url, model=llm_model, timeout=120)
        except Exception as exc:
            llm_error = str(exc)

    if not answer:
        answer = extractive_answer(question, results, llm_error=llm_error)

    return {
        "question": question,
        "answer": answer,
        "mode": mode,
        "llm_model": llm_model if use_llm and not llm_error else None,
        "llm_error": llm_error,
        "sources": [
            {
                "source_id": f"S{index}",
                "chunk_id": item.get("chunk_id"),
                "doc_type": item.get("doc_type"),
                "source_file": item.get("source_file"),
                "metadata": item.get("metadata", {}),
                "sources": item.get("sources", []),
                "rrf_score": item.get("rrf_score"),
            }
            for index, item in enumerate(results, start=1)
        ],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ask the local football RAG system.")
    parser.add_argument("question")
    parser.add_argument("--bm25-index", default="output/retrieval/bm25_index.json")
    parser.add_argument("--vector-dir", default="output/retrieval/vector-local")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-base-url", default=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    parser.add_argument("--llm-model", default=os.getenv("LOCAL_CHAT_MODEL", "qwen2.5:7b-instruct"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = answer_question(
        args.question,
        bm25_index_path=Path(args.bm25_index),
        vector_dir=Path(args.vector_dir),
        mode=args.mode,
        top_k=args.top_k,
        use_llm=not args.no_llm,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(payload["answer"])
        print("\nSources:")
        for source in payload["sources"]:
            print(f"- [{source['source_id']}] {source['chunk_id']} ({source['source_file']})")


if __name__ == "__main__":
    main()
