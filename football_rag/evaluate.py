from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .answer import answer_question
from .bm25 import normalize_text
from .retrieve import hybrid_search


DEFAULT_CASES = [
    {
        "id": "saka_goals",
        "question": "How many goals did Bukayo Saka score?",
        "expected_contains": ["Bukayo Saka", "16 goals"],
    },
    {
        "id": "league_winner",
        "question": "Which team won the Premier League 2023-24?",
        "expected_contains": ["Manchester City", "finished 1st"],
    },
    {
        "id": "man_city_transfer_spending",
        "question": "Manchester City transfer spending 2023 24",
        "expected_contains": ["Manchester City", "£212,168,000"],
    },
    {
        "id": "arsenal_points",
        "question": "How many points did Arsenal have in 2023-24?",
        "expected_contains": ["Arsenal", "89 points"],
    },
    {
        "id": "haaland_team",
        "question": "Which team did Erling Haaland play for?",
        "expected_contains": ["Erling Haaland", "Manchester City"],
    },
]


@dataclass
class EvalResult:
    case_id: str
    question: str
    hit: bool
    rank: int | None
    reciprocal_rank: float
    top_chunk_ids: list[str]
    answer: str | None


def load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_CASES
    return json.loads(path.read_text(encoding="utf-8"))


def text_matches(text: str, expected_terms: list[str]) -> bool:
    normalized_text = normalize_text(text)
    return all(normalize_text(term) in normalized_text for term in expected_terms)


def find_hit_rank(results: list[dict[str, Any]], expected_terms: list[str]) -> int | None:
    for rank, item in enumerate(results, start=1):
        searchable = " ".join(
            [
                str(item.get("chunk_id") or ""),
                str(item.get("doc_type") or ""),
                str(item.get("source_file") or ""),
                str(item.get("metadata") or ""),
                str(item.get("text") or ""),
            ]
        )
        if text_matches(searchable, expected_terms):
            return rank
    return None


def evaluate_case(
    case: dict[str, Any],
    *,
    bm25_index_path: Path,
    vector_dir: Path,
    mode: str,
    top_k: int,
    include_answers: bool,
) -> EvalResult:
    results = hybrid_search(
        case["question"],
        bm25_index_path=bm25_index_path,
        vector_dir=vector_dir,
        mode=mode,
        top_k=top_k,
    )
    rank = find_hit_rank(results, [str(term) for term in case["expected_contains"]])
    answer = None
    if include_answers:
        answer_payload = answer_question(
            case["question"],
            bm25_index_path=bm25_index_path,
            vector_dir=vector_dir,
            mode=mode,
            top_k=top_k,
            use_llm=False,
            llm_base_url="",
            llm_model="",
        )
        answer = answer_payload["answer"]

    return EvalResult(
        case_id=str(case["id"]),
        question=str(case["question"]),
        hit=rank is not None,
        rank=rank,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
        top_chunk_ids=[str(item.get("chunk_id")) for item in results],
        answer=answer,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate football RAG retrieval quality.")
    parser.add_argument("--cases", default=None, help="Optional JSON list of evaluation cases.")
    parser.add_argument("--bm25-index", default="output/retrieval/bm25_index.json")
    parser.add_argument("--vector-dir", default="output/retrieval/vector-local")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answers", action="store_true", help="Also generate no-LLM extractive answers.")
    parser.add_argument("--output", default="output/evaluation/eval_report.json")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases) if args.cases else None)
    results = [
        evaluate_case(
            case,
            bm25_index_path=Path(args.bm25_index),
            vector_dir=Path(args.vector_dir),
            mode=args.mode,
            top_k=args.top_k,
            include_answers=args.answers,
        )
        for case in cases
    ]

    hit_count = sum(1 for result in results if result.hit)
    report = {
        "mode": args.mode,
        "top_k": args.top_k,
        "case_count": len(results),
        "hit_at_k": hit_count / len(results) if results else 0.0,
        "mrr": sum(result.reciprocal_rank for result in results) / len(results) if results else 0.0,
        "results": [result.__dict__ for result in results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
