import json, os, re, sys, string
from collections import Counter
import numpy as np

QA_DIR     = "../data/qa"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

if "--qa-dir" in sys.argv:
    QA_DIR = sys.argv[sys.argv.index("--qa-dir") + 1]


def normalize_answer(text: str) -> str:
    """ Loại bỏ dấu câu, chuyển về lowercase, xóa stopwords và chuẩn hóa alias để tính EM/F1."""
    text = text.lower()
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\bman city\b", "manchester city", text)
    text = re.sub(r"\bman utd\b", "manchester united", text)
    text = re.sub(r"\bman united\b", "manchester united", text)
    text = re.sub(r"\bspurs\b", "tottenham", text)
    text = re.sub(r"\b(fc|afc)\b", " ", text)
    position_aliases = {
        "dc": "df", "dl": "df", "dr": "df", "cb": "df", "lb": "df", "rb": "df",
        "mc": "mf", "dm": "mf", "am": "mf", "cm": "mf", "ml": "mf", "mr": "mf",
        "lw": "mf", "rw": "mf",
        "cf": "fw", "st": "fw",
    }
    text = " ".join(position_aliases.get(tok, tok) for tok in text.split())
    return " ".join(text.split())


def get_tokens(text: str) -> list[str]:
    """Tách câu trả lời đã normalize thành danh sách token để tính F1."""
    return normalize_answer(text).split()


def exact_match(pred: str, refs: list[str]) -> float:
    """Trả về 1 nếu prediction khớp hoàn toàn với một reference sau normalize."""
    pred_n = normalize_answer(pred)
    return float(any(pred_n == normalize_answer(r) for r in refs))


def position_exact_match(pred: str, refs: list[str]) -> float:
    """Chấm đúng nếu prediction nằm trong tập vị trí reference."""
    pred_positions = {t for t in normalize_answer(pred).split() if t in {"gk", "df", "mf", "fw"}}
    if not pred_positions:
        return exact_match(pred, refs)
    for ref in refs:
        ref_positions = {t for t in normalize_answer(ref).split() if t in {"gk", "df", "mf", "fw"}}
        if pred_positions & ref_positions:
            return 1.0
    return 0.0


def token_f1(pred: str, ref: str) -> float:
    """Tính F1 token-level giữa một prediction và một reference."""
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
    """Lấy F1 cao nhất khi một câu hỏi có nhiều đáp án reference."""
    return max(token_f1(pred, r) for r in refs)


def answer_recall(pred: str, refs: list[str]) -> float:
    """Cho câu có nhiều đáp án (A; B; C), đo % đáp án được đề cập."""
    all_ans = []
    for r in refs:
        all_ans.extend([a.strip() for a in r.split(";") if a.strip()])
    if not all_ans:
        return 0.0
    pred_n = normalize_answer(pred)
    hits = sum(1 for a in all_ans if normalize_answer(a) in pred_n)
    return hits / len(all_ans)


def parse_refs(line: str) -> list[str]:
    """Tách một dòng reference thành nhiều đáp án bằng dấu chấm phẩy."""
    return [a.strip() for a in line.split(";") if a.strip()]


def load_lines(path: str) -> list[str]:
    """Đọc file text và bỏ các dòng bị rỗng."""
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def resolve_eval_paths(qa_dir: str) -> tuple[str, str, str, str]:
    """ Xác định đường dẫn đến reference answers, questions, metadata và thư mục IAA."""
    if os.path.exists(f"{qa_dir}/reference_answers.txt"):
        split_dir = qa_dir
        root_dir = os.path.dirname(qa_dir.rstrip("/"))
    else:
        root_dir = qa_dir
        split_dir = f"{qa_dir}/test"

    return (
        f"{split_dir}/reference_answers.txt",
        f"{split_dir}/questions.txt",
        f"{split_dir}/qa_pairs.json",
        root_dir,
    )

def evaluate_system(preds: list[str], refs:  list[list[str]], questions: list[str],metadata: list[dict] | None = None) -> dict:
    """Tính EM, F1 và Answer Recall cho một hệ thống dựa trên predictions và references."""
    assert len(preds) == len(refs), f"Length mismatch: {len(preds)} vs {len(refs)}"

    em_scores, f1_scores, rec_scores = [], [], []
    per_q = []

    for i, (pred, ref_list, q) in enumerate(zip(preds, refs, questions)):
        em  = exact_match(pred, ref_list)
        qtype = metadata[i].get("type", "unknown") if metadata else "unknown"
        if qtype == "player_position":
            em = position_exact_match(pred, ref_list)
        f1  = best_f1(pred, ref_list)
        rec = answer_recall(pred, ref_list)
        em_scores.append(em);  f1_scores.append(f1);  rec_scores.append(rec)

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

    # Phân loại đánh giá theo question type
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

    # Phân loại đánh giá theo difficulty
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


def cohens_kappa(ann1: list[str], ann2: list[str]) -> float:
    """Tính Cohen's Kappa cho hai annotator dựa trên câu trả lời đã normalize."""
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


def print_comparison(results: dict):
    """In bảng so sánh BM25/Dense theo overall, type và difficulty."""
    systems = list(results.keys())
    metric_col_width = 20
    value_col_width = 12
    print(f"\n{'═'*62}")
    print(f"  OVERALL RESULTS")
    print(f"{'─'*62}")
    print(f"  {'Metric':<{metric_col_width}}", end="")
    for s in systems:
        print(f"  {s[:value_col_width]:>{value_col_width}}", end="")
    print()
    print(f"{'─'*62}")
    for metric in ["EM", "F1", "AnswerRecall"]:
        print(f"  {metric:<{metric_col_width}}", end="")
        for s in systems:
            v = results[s]["overall"][metric]
            print(f"  {v:>{value_col_width}.4f}", end="")
        print()
    print(f"{'═'*62}")

    # Phân loại đánh giá theo question type
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

    # Phân loại đánh giá theo difficulty
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


def main():
    """Load predictions/reference, tính metric, IAA."""

    # Xác định đường dẫn đến reference answers, questions, metadata và thư mục IAA
    ref_path, q_path, meta_path, iaa_dir = resolve_eval_paths(QA_DIR)

    refs_raw  = load_lines(ref_path)
    questions = load_lines(q_path)
    refs      = [parse_refs(r) for r in refs_raw]

    metadata = None
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)

    print(f"\nTest set: {len(questions)} questions\n")

    # 
    system_files = {
        "BM25":  f"{OUTPUT_DIR}/system_output_bm25.txt",
        "Dense": f"{OUTPUT_DIR}/system_output_dense.txt",
    }
    results = {}

    for sys_name, pred_path in system_files.items():
        preds = load_lines(pred_path)
        preds = preds[:len(refs)]
        while len(preds) < len(refs):
            preds.append("")
        print(f"  Evaluating {sys_name}...")
        results[sys_name] = evaluate_system(preds, refs, questions, metadata)
        ov = results[sys_name]["overall"]
        print(f"    EM={ov['EM']:.4f}  F1={ov['F1']:.4f}  Recall={ov['AnswerRecall']:.4f}")

    print_comparison(results)

    # Tính Cohen's Kappa nếu có dữ liệu IAA
    kappa_result = {}
    ann1_path = f"{iaa_dir}/iaa_annotator1.txt"
    ann2_path = f"{iaa_dir}/iaa_annotator2.txt"

    # Nếu có dữ liệu IAA, tính Cohen's Kappa và thống kê agreement
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

    # Lưu báo cáo chi tiết dưới dạng JSON
    report = {
        "dataset":     "EPL 2023-24 Transfers & Match Data",
        "test_size":   len(questions),
        "systems":     results,
        "iaa":         kappa_result,
    }
    report_path = f"{OUTPUT_DIR}/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Lưu báo cáo tóm tắt dưới dạng text
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


if __name__ == "__main__":
    main()
