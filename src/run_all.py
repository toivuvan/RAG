#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC = PROJECT_ROOT / "src"


def run_step(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    subprocess.run(command, cwd=str(cwd), env=merged_env, check=True)


def parse_args() -> argparse.Namespace:
    """Định nghĩa các tham số CLI để chạy toàn bộ hoặc từng phần pipeline."""
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
    parser.add_argument(
        "--qa-dir",
        default=None,
        help="Optional QA directory passed to pipeline/evaluate. If omitted, scripts use their own default ../data/qa.",
    )
    return parser.parse_args()


def main() -> None:
    """Điều phối thứ tự chạy: chunker -> embedding -> generate_qa -> pipeline -> evaluate."""
    args = parse_args()
    python = args.python

    hf_env = {}
    if args.offline:
        hf_env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

    qa_dir = None
    if args.qa_dir:
        qa_path = Path(args.qa_dir)
        if not qa_path.is_absolute():
            project_relative = PROJECT_ROOT / qa_path
            qa_path = project_relative if project_relative.exists() else SRC / qa_path
        qa_dir = str(qa_path)

    if not args.only_eval:
        if not args.skip_chunking:
            run_step([python, "chunker.py"], SRC)

        if not args.skip_indexing:
            run_step([python, "embedding.py"], SRC, hf_env)

        if not args.skip_qa:
            run_step([python, "generate_qa.py"], SRC)

        if not args.skip_rag:
            rag_cmd = [python, "pipeline.py", "--approach", args.approach]
            if qa_dir:
                rag_cmd.extend(["--qa-dir", qa_dir])
            run_step(rag_cmd, SRC, hf_env)

    eval_cmd = [python, "evaluate.py"]
    if qa_dir:
        eval_cmd.extend(["--qa-dir", qa_dir])
    if args.iaa:
        eval_cmd.append("--iaa")
    run_step(eval_cmd, SRC)



if __name__ == "__main__":
    main()
