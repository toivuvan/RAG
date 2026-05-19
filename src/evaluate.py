"""
STEP 5 — EVALUATION
=====================
Tính Exact Match, F1, Answer Recall (chuẩn SQuAD) cho cả 2 approach.
So sánh BM25 vs Dense và phân tích theo question type.
Tính Cohen's Kappa cho IAA nếu có file annotator.

Output:
  outputs/evaluation_report.json    ← full metrics
  outputs/evaluation_summary.txt    ← human-readable table

Chạy:
  python src/05_evaluate.py
  python src/05_evaluate.py --iaa    (+ tính Kappa nếu có 2 annotator files)
"""

import json, os, re, sys, string
from collections import Counter, defaultdict
import numpy as np

QA_DIR     = "../data/qa"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  METRIC FUNCTIONS

def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation & articles."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Normalize football-specific aliases
    text = re.sub(r"\bman city\b", "manchester city", text)
    text = re.sub(r"\bman utd\b", "manchester united", text)
    text = re.sub(r"\bman united\b", "manchester united", text)
    text = re.sub(r"\bspurs\b", "tottenham", text)
    return " ".join(text.split())


def get_tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def exact_match(pred: str, refs: list[str]) -> float:
    pred_n = normalize_answer(pred)
    return float(any(pred_n == normalize_answer(r) for r in refs))


def token_f1(pred: str, ref: str) -> float:
    pred_toks = get_tokens(pred)
    ref_toks  = get_tokens(ref)
    if not pred_toks or not ref_toks:
        return float(pred_toks == ref_toks)
    common  = Counter(pred_toks) & Counter(ref_toks)
    n_match = sum(common.values())
    if n_match == 0:
        return 0.0
    precision = n_match / len(pred_toks)
    recall    = n_match / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, refs: list[str]) -> float:
    return max(token_f1(pred, r) for r in refs)


def answer_recall(pred: str, refs: list[str]) -> float:
    """
    Cho câu có nhiều đáp án (A; B; C), đo % đáp án được đề cập.
    """
    all_ans = []
    for r in refs:
        all_ans.extend([a.strip() for a in r.split(";") if a.strip()])
    if not all_ans:
        return 0.0
    pred_n = normalize_answer(pred)
    hits = sum(1 for a in all_ans if normalize_answer(a) in pred_n)
    return hits / len(all_ans)


def parse_refs(line: str) -> list[str]:
    return [a.strip() for a in line.split(";") if a.strip()]

#  LOAD PREDICTIONS & REFERENCES

def load_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def evaluate_system(
    preds: list[str],
    refs:  list[list[str]],
    questions: list[str],
    metadata: list[dict] | None = None,
) -> dict:
    assert len(preds) == len(refs), f"Length mismatch: {len(preds)} vs {len(refs)}"

    em_scores, f1_scores, rec_scores = [], [], []
    per_q = []

    for i, (pred, ref_list, q) in enumerate(zip(preds, refs, questions)):
        em  = exact_match(pred, ref_list)
        f1  = best_f1(pred, ref_list)
        rec = answer_recall(pred, ref_list)
        em_scores.append(em);  f1_scores.append(f1);  rec_scores.append(rec)

        qtype = metadata[i].get("type", "unknown") if metadata else "unknown"
        diff  = metadata[i].get("difficulty", "")  if metadata else ""
        per_q.append({
            "id": i+1, "question": q, "prediction": pred,
            "references": ref_list,
            "EM": em, "F1": round(f1,4), "Recall": round(rec,4),
            "type": qtype, "difficulty": diff,
        })

    overall = {
        "EM":           round(float(np.mean(em_scores)), 4),
        "F1":           round(float(np.mean(f1_scores)), 4),
        "AnswerRecall": round(float(np.mean(rec_scores)), 4),
        "n":            len(preds),
    }

    # By type
    by_type = {}
    if metadata:
        types = {m.get("type","?") for m in metadata}
        for t in sorted(types):
            idx = [i for i, m in enumerate(metadata) if m.get("type")==t]
            by_type[t] = {
                "EM":   round(float(np.mean([em_scores[i]  for i in idx])), 4),
                "F1":   round(float(np.mean([f1_scores[i]  for i in idx])), 4),
                "Recall": round(float(np.mean([rec_scores[i] for i in idx])), 4),
                "n":    len(idx),
            }

    # By difficulty
    by_diff = {}
    if metadata:
        for diff in ("easy", "medium", "hard"):
            idx = [i for i, m in enumerate(metadata) if m.get("difficulty")==diff]
            if idx:
                by_diff[diff] = {
                    "EM":   round(float(np.mean([em_scores[i]  for i in idx])), 4),
                    "F1":   round(float(np.mean([f1_scores[i]  for i in idx])), 4),
                    "n":    len(idx),
                }

    return {"overall": overall, "by_type": by_type,
            "by_difficulty": by_diff, "per_question": per_q}

#  COHEN'S KAPPA — IAA

def cohens_kappa(ann1: list[str], ann2: list[str]) -> float:
    n = len(ann1)
    assert n == len(ann2)
    agree = sum(1 for a, b in zip(ann1, ann2)
                if normalize_answer(a) == normalize_answer(b))
    po = agree / n
    unique_vals = list({normalize_answer(x) for x in ann1 + ann2})
    pe = sum(
        (ann1.count(v) / n) * (ann2.count(v) / n)
        for v in unique_vals
    )
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0

#  PRETTY PRINT

def print_comparison(results: dict):
    systems = list(results.keys())
    print(f"\n{'═'*62}")
    print(f"  OVERALL RESULTS")
    print(f"{'─'*62}")
    print(f"  {'Metric':<20}", end="")
    for s in systems:
        print(f"  {s[:12]:>12}", end="")
    print()
    print(f"{'─'*62}")
    for metric in ["EM", "F1", "AnswerRecall"]:
        vals = [results[s]["overall"][metric] for s in systems]
        best = max(vals)
        print(f"  {metric:<20}", end="")
        for v in vals:
            marker = " ✓" if v == best and len(systems) > 1 else "  "
            print(f"  {v:>10.4f}{marker}", end="")
        print()
    print(f"{'═'*62}")

    # By type (show F1)
    all_types = sorted({t for s in systems
                        for t in results[s].get("by_type", {})})
    if all_types:
        print(f"\n  F1 by Question Type:")
        print(f"  {'Type':<28}", end="")
        for s in systems: print(f"  {s[:10]:>10}", end="")
        print(f"  {'n':>5}")
        print(f"  {'─'*60}")
        for t in all_types:
            n = results[systems[0]]["by_type"].get(t, {}).get("n", 0)
            print(f"  {t:<28}", end="")
            for s in systems:
                v = results[s]["by_type"].get(t, {}).get("F1", "-")
                print(f"  {str(v):>10}", end="")
            print(f"  {n:>5}")

    # By difficulty
    print(f"\n  F1 by Difficulty:")
    print(f"  {'Difficulty':<12}", end="")
    for s in systems: print(f"  {s[:10]:>10}", end="")
    print()
    for diff in ("easy", "medium", "hard"):
        print(f"  {diff:<12}", end="")
        for s in systems:
            v = results[s].get("by_difficulty", {}).get(diff, {}).get("F1", "-")
            print(f"  {str(v):>10}", end="")
        print()

#  MAIN

def main():
    print("=" * 62)
    print("  STEP 5: EVALUATION")
    print("=" * 62)

    # Load references & questions
    ref_path = f"{QA_DIR}/test/reference_answers.txt"
    q_path   = f"{QA_DIR}/test/questions.txt"
    meta_path= f"{QA_DIR}/test/qa_pairs.json"

    refs_raw  = load_lines(ref_path)
    questions = load_lines(q_path)
    refs      = [parse_refs(r) for r in refs_raw]

    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    print(f"\nTest set: {len(questions)} questions\n")

    # ── Evaluate available systems
    system_files = {
        "BM25":  f"{OUTPUT_DIR}/system_output_bm25.txt",
        "Dense": f"{OUTPUT_DIR}/system_output_dense.txt",
    }
    results = {}

    for sys_name, pred_path in system_files.items():
        if not os.path.exists(pred_path):
            print(f"  [SKIP] {sys_name}: {pred_path} not found")
            continue
        preds = load_lines(pred_path)
        if len(preds) < len(refs):
            print(f"  [WARNING] {sys_name}: only {len(preds)} predictions found; missing {len(refs)-len(preds)} will be scored as blank.")
        elif len(preds) > len(refs):
            print(f"  [WARNING] {sys_name}: {len(preds)} predictions found but only {len(refs)} references; extra predictions will be ignored.")
        # Align lengths
        preds = preds[:len(refs)]
        while len(preds) < len(refs):
            preds.append("")
        print(f"  Evaluating {sys_name}...")
        results[sys_name] = evaluate_system(preds, refs, questions, metadata)
        ov = results[sys_name]["overall"]
        print(f"    EM={ov['EM']:.4f}  F1={ov['F1']:.4f}  Recall={ov['AnswerRecall']:.4f}")

    print_comparison(results)

    # ── IAA (Cohen's Kappa)
    kappa_result = {}
    ann1_path = f"{QA_DIR}/iaa_annotator1.txt"
    ann2_path = f"{QA_DIR}/iaa_annotator2.txt"

    if "--iaa" in sys.argv and os.path.exists(ann1_path) and os.path.exists(ann2_path):
        ann1 = load_lines(ann1_path)
        ann2 = load_lines(ann2_path)
        min_len = min(len(ann1), len(ann2))
        kappa = cohens_kappa(ann1[:min_len], ann2[:min_len])
        agree_raw = sum(1 for a, b in zip(ann1[:min_len], ann2[:min_len])
                       if normalize_answer(a) == normalize_answer(b))
        kappa_result = {
            "kappa": kappa,
            "n_questions": min_len,
            "agreements": agree_raw,
            "agreement_pct": round(agree_raw / min_len * 100, 1),
            "interpretation": (
                "Almost perfect" if kappa > 0.8 else
                "Substantial"    if kappa > 0.6 else
                "Moderate"       if kappa > 0.4 else
                "Fair"           if kappa > 0.2 else "Slight"
            ),
        }
        print(f"\n  IAA — Cohen's Kappa: {kappa} ({kappa_result['interpretation']})")
        print(f"    Agreement: {agree_raw}/{min_len} ({kappa_result['agreement_pct']}%)")
    else:
        kappa_result = {
            "note": (
                "To compute IAA: have 2 annotators answer data/qa/iaa_subset.json, "
                "save answers to data/qa/iaa_annotator1.txt and iaa_annotator2.txt, "
                "then rerun with --iaa flag."
            )
        }
        print(f"\n  IAA: {kappa_result['note']}")

    # ── Save full report
    report = {
        "dataset":     "EPL 2023-24 Transfers & Match Data",
        "test_size":   len(questions),
        "systems":     results,
        "iaa":         kappa_result,
    }
    report_path = f"{OUTPUT_DIR}/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Save human-readable summary
    summary_path = f"{OUTPUT_DIR}/evaluation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EPL 2023-24 RAG System — Evaluation Summary\n")
        f.write("=" * 55 + "\n\n")
        for sname, sres in results.items():
            ov = sres["overall"]
            f.write(f"System: {sname}\n")
            f.write(f"  EM={ov['EM']:.4f}  F1={ov['F1']:.4f}  AnswerRecall={ov['AnswerRecall']:.4f}\n")
            f.write(f"  N={ov['n']}\n\n")
            f.write("  By Question Type (F1):\n")
            for t, m in sorted(sres["by_type"].items()):
                f.write(f"    {t:<28} F1={m['F1']:.4f} (n={m['n']})\n")
            f.write("\n")
        f.write(f"IAA: {json.dumps(kappa_result, indent=2)}\n")

    print(f"\n{'='*62}")
    print(f"  Reports saved:")
    print(f"    {report_path}")
    print(f"    {summary_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
