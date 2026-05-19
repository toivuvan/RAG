#!/usr/bin/env python3
"""
Chạy toàn bộ chương trình EPL RAG.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR / "chunker.py").exists() else SCRIPT_DIR
SRC = PROJECT_ROOT / "src"
DATA_CHUNKS = PROJECT_ROOT / "data" / "chunks"
OUTPUTS = PROJECT_ROOT / "outputs"

CHUNK_FILES = [
    "all_players.jsonl",
    "all_teams.jsonl",
    "all_matches.jsonl",
    "transfers_chunks.jsonl",
    "schedule_chunks.jsonl",
]


def run_step(name: str, command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"\n{'=' * 70}", flush=True)
    print(f"RUNNING: {name}", flush=True)
    print(f"CMD    : {' '.join(command)}", flush=True)
    print(f"CWD    : {cwd}", flush=True)
    print(f"{'=' * 70}", flush=True)

    start = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    subprocess.run(command, cwd=str(cwd), env=merged_env, check=True)
    print(f"\nDONE: {name} ({time.time() - start:.1f}s)", flush=True)


def validate_chunks() -> None:
    missing = []

    for filename in CHUNK_FILES:
        path = DATA_CHUNKS / filename
        if not path.exists():
            missing.append(str(path))

    if missing:
        raise FileNotFoundError(
            "Chunking finished, but these expected files were not found:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    print(f"\nValidated {len(CHUNK_FILES)} chunk files in {DATA_CHUNKS}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EPL 2023-24 RAG pipeline.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run pipeline scripts. Default: current interpreter.",
    )
    parser.add_argument(
        "--approach",
        choices=["both", "bm25", "dense"],
        default="both",
        help="Retrieval approach passed to src/pipeline.py.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for cached Hugging Face models.",
    )
    parser.add_argument("--skip-chunking", action="store_true", help="Skip src/chunker.py.")
    parser.add_argument("--skip-indexing", action="store_true", help="Skip src/embedding.py.")
    parser.add_argument("--skip-qa", action="store_true", help="Skip src/generate_qa.py.")
    parser.add_argument("--skip-rag", action="store_true", help="Skip src/pipeline.py.")
    parser.add_argument("--only-eval", action="store_true", help="Only run src/evaluate.py.")
    parser.add_argument("--iaa", action="store_true", help="Pass --iaa to src/evaluate.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = args.python

    hf_env = {}
    if args.offline:
        hf_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    if not args.only_eval:
        if not args.skip_chunking:
            run_step("1. Chunk raw data", [python, "src/chunker.py"], PROJECT_ROOT)
            validate_chunks()

        if not args.skip_indexing:
            run_step("2. Build BM25 and FAISS indexes", [python, "src/embedding.py"], PROJECT_ROOT, hf_env)

        if not args.skip_qa:
            run_step("3. Generate train/test QA", [python, "generate_qa.py"], SRC)

        if not args.skip_rag:
            run_step(
                "4. Run RAG systems",
                [python, "pipeline.py", "--approach", args.approach],
                SRC,
                hf_env,
            )

    eval_cmd = [python, "evaluate.py"]
    if args.iaa:
        eval_cmd.append("--iaa")
    run_step("5. Evaluate outputs", eval_cmd, SRC)

    print(f"\nAll requested steps completed. Reports are in: {OUTPUTS}")


if __name__ == "__main__":
    main()
