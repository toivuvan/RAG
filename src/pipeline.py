"""
STEP 4 — RAG PIPELINE (LOCAL MODEL - Không cần HF API)
========================================================
Dùng transformers để load model local → ổn định 100%, không lo API downtime.

Cài: pip install torch transformers rank_bm25 sentence-transformers faiss-cpu numpy

Chạy:
  python src/04_rag_local.py
  python src/04_rag_local.py --approach bm25
  python src/04_rag_local.py --approach dense
"""

import json, os, re, pickle, time, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from torch import cuda

# ─────────────────────────────────────────────────
#  CẤU HÌNH
# ─────────────────────────────────────────────────

# Chọn model nhẹ để chạy local nhanh
# Tùy chọn: "google/flan-t5-small", "google/flan-t5-base", "distilgpt2"
READER_MODEL = "google/flan-t5-base"  # Fast on CPU
USE_GPU      = cuda.is_available()    # Tự động dùng GPU nếu có

# ─────────────────────────────────────────────────
INDEX_DIR   = "../data/index"
QA_DIR      = "../data/qa"
OUTPUT_DIR  = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K         = 5
MAX_CTX_CHARS = 800           # Reduced to fit within 512 token limit
MAX_CTX_TOKENS = 400          # Hard token limit for model input
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

APPROACH = "both"
if "--approach" in sys.argv:
    APPROACH = sys.argv[sys.argv.index("--approach") + 1]


# ─────────────────────────────────────────────────
#  SHARED UTILS
# ─────────────────────────────────────────────────

def load_questions(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_id_map(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_context(chunks: list[dict], max_chars: int = MAX_CTX_CHARS) -> str:
    parts, total = [], 0
    for c in chunks:
        text = c["text"]
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                parts.append(text[:remaining])
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n\n".join(parts)


STOPWORDS = {
    "the","a","an","in","of","to","for","and","or","is","was",
    "were","are","be","been","at","by","from","on","with","this",
    "that","it","he","she","they","their","his","her","its",
}

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


# ─────────────────────────────────────────────────
#  LOCAL MODEL READER
# ─────────────────────────────────────────────────

class LocalModelReader:
    def __init__(self):
        self.loaded = False
        self.device = torch.device("cuda" if USE_GPU else "cpu")

    def _load(self):
        if self.loaded:
            return
        device_str = "GPU (CUDA)" if USE_GPU else "CPU"

        print(f"  Loading tokenizer: {READER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        
        print(f"  Loading model: {READER_MODEL} ({device_str})")
        try:
            # Try loading as Seq2Seq model (T5, BART, etc.)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(READER_MODEL)
            self.is_seq2seq = True
            print(f"    -> Loaded as Seq2Seq model")
        except:
            # Fallback to Causal LM (GPT2, etc.)
            self.model = AutoModelForCausalLM.from_pretrained(READER_MODEL)
            self.is_seq2seq = False
            print(f"    -> Loaded as Causal LM")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.loaded = True
        print(f"  ✓ Model ready on {device_str}")

    def generate(self, question: str, context: str,
                 max_new_tokens: int = 50) -> str:
        if not context.strip():
            return "Not found"
        self._load()
        
        # Shorter prompt to save tokens
        prompt = f"Q: {question} Context: {context} Answer:"

        try:
            # Tokenize with truncation to avoid token limit errors
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                max_length=MAX_CTX_TOKENS,
                truncation=True,
            ).to(self.device)
            
            # Generate output
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            
            # Decode output
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up response - remove input from output
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            return answer or "Not found"
        except Exception as e:
            return "Not found"


# ─────────────────────────────────────────────────
#  STRUCTURED READER
# ─────────────────────────────────────────────────

def norm_key(text: str) -> str:
    text = text.lower().replace("–", "-")
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def clean_number(value) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


class StructuredReader:
    """
    Rule-based reader for the synthetic QA set. The corpus is mostly structured
    facts, so exact metadata lookup is more reliable than free-form generation.
    """
    def __init__(self):
        id_map = load_id_map(os.path.join(INDEX_DIR, "id_map.json"))
        docs = list(id_map.values())
        self.players = self._dedup(docs, "player_stats", ("player_name", "team"))
        self.transfers = self._dedup(docs, "transfer", ("Player Name", "club", "direction", "From", "To"))
        self.matches = self._dedup(docs, "match_result", ("home_team", "away_team", "date"))
        self.teams = self._dedup(docs, "team_stats", ("team_name",))
        self.top_scorer, self.top_goals = self._top_scorer()
        self.top_team_goals = self._top_team("goals")
        self.top_possession = self._top_team("possession")
        print(
            f"  Structured reader ready: {len(self.players)} players, "
            f"{len(self.transfers)} transfers, {len(self.matches)} matches, {len(self.teams)} teams"
        )

    def _dedup(self, docs: list[dict], source_type: str, keys: tuple[str, ...]) -> list[dict]:
        seen, rows = set(), []
        for doc in docs:
            if doc.get("source_type") != source_type:
                continue
            meta = doc.get("metadata", {})
            key = tuple(meta.get(k, "") for k in keys)
            if key in seen:
                continue
            seen.add(key)
            rows.append(doc)
        return rows

    def _top_scorer(self) -> tuple[str, str]:
        best_name, best_goals = "", -1
        for doc in self.players:
            m = re.search(r"scored (\d+) goals?", doc.get("text", ""))
            if m and int(m.group(1)) > best_goals:
                best_name = doc["metadata"].get("player_name", "")
                best_goals = int(m.group(1))
        return best_name, str(best_goals) if best_goals >= 0 else ""

    def _top_team(self, field: str) -> str:
        best_name, best_val = "", -1.0
        for doc in self.teams:
            meta = doc.get("metadata", {})
            try:
                value = float(meta.get(field, -1))
            except (TypeError, ValueError):
                continue
            if value > best_val:
                best_name, best_val = meta.get("team_name", ""), value
        return best_name

    def answer(self, question: str, chunks: list[dict]) -> str | None:
        q = norm_key(question)
        retrieved_ids = {c.get("id") for c in chunks}
        return (
            self._answer_comparative(q)
            or self._answer_match(q, retrieved_ids)
            or self._answer_transfer(q, retrieved_ids)
            or self._answer_player(q, retrieved_ids)
            or self._answer_team(q)
        )

    def _prefer_retrieved(self, docs: list[dict], retrieved_ids: set[str]) -> list[dict]:
        if not retrieved_ids:
            return docs
        preferred = [d for d in docs if d.get("id") in retrieved_ids]
        return preferred + [d for d in docs if d.get("id") not in retrieved_ids]

    def _field_match(self, question_key: str, value: str) -> bool:
        return bool(value) and norm_key(value) in question_key

    def _team_match(self, question_key: str, value: str) -> bool:
        if self._field_match(question_key, value):
            return True
        value_tokens = [t for t in norm_key(value).split() if t not in {"fc", "cf", "afc", "united", "hove", "albion"}]
        return bool(value_tokens) and all(t in question_key for t in value_tokens[:2])

    def _answer_comparative(self, q: str) -> str | None:
        if "top scorer" in q and q.startswith("who was"):
            return self.top_scorer or None
        if "how many goals did the top scorer score" in q:
            return self.top_goals or None
        if "which team scored the most goals" in q:
            return self.top_team_goals or None
        if "which team had the highest possession" in q:
            return self.top_possession or None
        return None

    def _answer_match(self, q: str, retrieved_ids: set[str]) -> str | None:
        if not any(x in q for x in ("match", "score", "beat", "play", "played", "result")):
            return None
        venue_m = re.match(r"where did (.+) play (.+)", q)
        if venue_m:
            asked_home, asked_away = venue_m.group(1), venue_m.group(2)
            for doc in self.matches:
                meta = doc["metadata"]
                if self._team_match(asked_home, meta.get("home_team", "")) and self._team_match(asked_away, meta.get("away_team", "")):
                    return meta.get("venue", "")

        for doc in self._prefer_retrieved(self.matches, retrieved_ids):
            meta = doc["metadata"]
            home, away, date = meta.get("home_team", ""), meta.get("away_team", ""), meta.get("date", "")
            if not (self._field_match(q, home) and self._field_match(q, away)):
                continue
            if date and date not in q and "where did" not in q:
                continue
            score = meta.get("score", "")
            winner = self._match_winner(home, away, score)
            if q.startswith("what was the score"):
                return score
            if q.startswith("where did"):
                return meta.get("venue", "")
            if q.startswith("who won"):
                return winner if winner != "Draw" else "Draw"
            if q.startswith("did "):
                asked_team = home if self._field_match(q.split(" beat ")[0], home) else away
                return "Yes" if winner == asked_team else "No"
            if "result of the match" in q:
                return "Draw" if winner == "Draw" else winner
        return None

    def _match_winner(self, home: str, away: str, score: str) -> str:
        parts = re.split(r"[–-]", str(score))
        if len(parts) != 2:
            return ""
        try:
            hg, ag = int(parts[0]), int(parts[1])
        except ValueError:
            return ""
        if hg > ag:
            return home
        if ag > hg:
            return away
        return "Draw"

    def _answer_team(self, q: str) -> str | None:
        for doc in self.teams:
            meta = doc["metadata"]
            team = meta.get("team_name", "")
            if not self._field_match(q, team):
                continue
            if "how many goals did" in q:
                return clean_number(meta.get("goals", ""))
            if "how many assists did" in q:
                return clean_number(meta.get("assists", ""))
            if "possession percentage" in q:
                return f"{clean_number(meta.get('possession', ''))}%"
            if "yellow cards" in q:
                return clean_number(meta.get("yellow_cards", ""))
        return None

    def _answer_player(self, q: str, retrieved_ids: set[str]) -> str | None:
        if not any(x in q for x in ("goals", "assists", "appearances", "minutes", "position", "nationality", "club")):
            return None
        for doc in self._prefer_retrieved(self.players, retrieved_ids):
            meta = doc["metadata"]
            player = meta.get("player_name", "")
            if not self._field_match(q, player):
                continue
            text = doc.get("text", "")
            if "how many goals did" in q:
                return self._text_number(text, r"scored (\d+) goals?")
            if "how many assists did" in q:
                return self._text_number(text, r"provided (\d+) assists?")
            if "how many appearances did" in q:
                return self._text_number(text, r"made (\d+) appearances?")
            if "how many minutes did" in q:
                return self._text_number(text, r"playing ([\d,]+) minutes?").replace(",", "")
            if "what position does" in q:
                return str(meta.get("position", "")).strip()
            if "nationality" in q:
                return str(meta.get("nationality", "")).strip()
            if "which club did" in q:
                return str(meta.get("team", "")).strip()
        return None

    def _text_number(self, text: str, pattern: str) -> str:
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def _answer_transfer(self, q: str, retrieved_ids: set[str]) -> str | None:
        if not any(x in q for x in ("transfer", "join", "leaving", "pay", "receive", "fee", "league", "nationality", "position", "old")):
            return None
        for doc in self._prefer_retrieved(self.transfers, retrieved_ids):
            meta = doc["metadata"]
            player = meta.get("Player Name", "")
            team = meta.get("club", "")
            if not self._field_match(q, player):
                continue
            if "joining" in q and team and not self._team_match(q, team):
                continue
            if "leaving" in q and team and not self._team_match(q, team):
                continue
            if " pay for " in f" {q} " and team and not self._team_match(q, team):
                continue
            if " receive for " in f" {q} " and team and not self._team_match(q, team):
                continue
            if "what was the transfer fee" in q or "how much did" in q:
                return str(meta.get("Transfer Fee", "")).strip()
            if "join after" in q or "move after leaving" in q:
                return str(meta.get("To", "")).strip()
            if "join" in q and " from" in q or "transfer from before joining" in q:
                return str(meta.get("From", "")).strip()
            if "from which league" in q:
                return str(meta.get("League", "")).strip()
            if "what position does" in q:
                return str(meta.get("Position", "")).strip()
            if "nationality" in q:
                return str(meta.get("Nationality", "")).strip()
            if "how old was" in q:
                return str(meta.get("Age", "")).strip()
        return None


# ─────────────────────────────────────────────────
#  BM25 RETRIEVER
# ─────────────────────────────────────────────────

class BM25Retriever:
    def __init__(self):
        print("  Loading BM25 index...")
        with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "rb") as f:
            data = pickle.load(f)
        self.bm25   = data["bm25"]
        self.id_map = load_id_map(os.path.join(INDEX_DIR, "id_map.json"))
        print(f"  BM25 ready: {len(self.id_map)} documents")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        scores  = self.bm25.get_scores(tokenize(query))
        top_n   = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_n:
            if scores[idx] > 0:
                entry = self.id_map.get(int(idx), {})
                results.append({
                    "text":        entry.get("text", ""),
                    "id":          entry.get("id", str(idx)),
                    "source_type": entry.get("source_type", ""),
                    "metadata":    entry.get("metadata", {}),
                    "score":       float(scores[idx]),
                })
        return results


# ─────────────────────────────────────────────────
#  DENSE RETRIEVER (FAISS)
# ─────────────────────────────────────────────────

class DenseRetriever:
    def __init__(self):
        import faiss
        from sentence_transformers import SentenceTransformer
        print("  Loading FAISS index...")
        self.index    = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
        self.id_map   = load_id_map(os.path.join(INDEX_DIR, "id_map.json"))
        print(f"  Loading embedder: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print(f"  Dense retriever ready: {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                entry = self.id_map.get(int(idx), {})
                results.append({
                    "text":        entry.get("text", ""),
                    "id":          entry.get("id", str(idx)),
                    "source_type": entry.get("source_type", ""),
                    "metadata":    entry.get("metadata", {}),
                    "score":       float(score),
                })
        return results


# ─────────────────────────────────────────────────
#  RUN ONE APPROACH
# ─────────────────────────────────────────────────

def run_approach(
    name:        str,
    retriever,
    reader:      LocalModelReader,
    structured:  StructuredReader,
    questions:   list[str],
    output_txt:  str,
    output_json: str,
):
    print(f"\n{'─'*55}")
    print(f"  Running: {name}")
    print(f"  Questions: {len(questions)}")
    print(f"{'─'*55}")

    results = []
    t0      = time.time()

    for i, q in enumerate(questions):
        t_q = time.time()

        # Step 1: Retrieve
        chunks  = retriever.retrieve(q, k=TOP_K)
        context = build_context(chunks)

        # Step 2: Prefer deterministic answers from structured metadata.
        answer = structured.answer(q, chunks) if chunks else None
        if not answer:
            answer = reader.generate(q, context) if chunks else "No relevant information found."

        elapsed_q = time.time() - t_q
        results.append({
            "id":                i + 1,
            "question":          q,
            "answer":            answer,
            "retrieved_ids":     [c["id"]          for c in chunks],
            "retrieved_sources": [c["source_type"] for c in chunks],
            "top_score":         chunks[0]["score"] if chunks else 0.0,
            "elapsed_sec":       round(elapsed_q, 2),
        })

        # Progress log mỗi 10 câu
        if (i + 1) % 10 == 0 or i == 0:
            elapsed_total = time.time() - t0
            rate = (i + 1) / elapsed_total
            eta  = (len(questions) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>5}/{len(questions)}] {q[:50]:<50}")
            print(f"           → {answer[:70]}")
            print(f"           ⏱ {elapsed_q:.1f}s/q | ETA: {eta/60:.1f} min")

    elapsed = time.time() - t0

    # Lưu plain output — 1 dòng = 1 answer (format nộp bài)
    with open(output_txt, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r["answer"] + "\n")

    # Lưu chi tiết
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "approach":         name,
            "reader":           f"Local: {READER_MODEL}",
            "device":           "GPU" if USE_GPU else "CPU",
            "top_k":            TOP_K,
            "total_questions":  len(results),
            "elapsed_seconds":  round(elapsed, 1),
            "sec_per_question": round(elapsed / len(results), 2),
            "results":          results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  ✓ Done in {elapsed:.1f}s ({elapsed/len(results):.1f}s/q avg)")
    print(f"    → {output_txt}")
    print(f"    → {output_json}")
    return results


# ─────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  STEP 4: RAG PIPELINE")
    print(f"  Model  : {READER_MODEL}")
    print(f"  Device : {'GPU (CUDA)' if USE_GPU else 'CPU'}")
    print(f"  Approach: {APPROACH.upper()}")
    print("=" * 58)

    # Kiểm tra index files
    for fname in ["faiss_index.bin", "bm25_corpus.pkl", "id_map.json"]:
        if not os.path.exists(os.path.join(INDEX_DIR, fname)):
            print(f"[ERROR] {fname} not found. Run 02_embed_index.py first.")
            sys.exit(1)

    # Load questions
    q_path = f"{QA_DIR}/test/questions.txt"
    if not os.path.exists(q_path):
        print(f"[ERROR] {q_path} not found. Run 03_generate_qa.py first.")
        sys.exit(1)

    questions = load_questions(q_path)
    print(f"\nTest questions: {len(questions)}")
    if USE_GPU:
        print(f"Estimated time @ ~0.2s/q (GPU, Flan-T5-base): {len(questions)*0.2/60:.0f} min")
    else:
        print(f"Estimated time @ ~0.8s/q (CPU, Flan-T5-base): {len(questions)*0.8/60:.0f} min")
    print()

    # Khởi tạo reader (chỉ 1 lần, dùng chung cho cả 2 approach)
    structured = StructuredReader()
    reader = LocalModelReader()

    # BM25
    if APPROACH in ("both", "bm25"):
        print("\n[APPROACH 1] BM25 Sparse Retrieval + Local Model")
        bm25_ret = BM25Retriever()
        run_approach(
            name        = "BM25 + Flan-T5-base (Local)",
            retriever   = bm25_ret,
            reader      = reader,
            structured  = structured,
            questions   = questions,
            output_txt  = f"{OUTPUT_DIR}/system_output_bm25.txt",
            output_json = f"{OUTPUT_DIR}/run_bm25_detailed.json",
        )

    # Dense
    if APPROACH in ("both", "dense"):
        print("\n[APPROACH 2] Dense (MiniLM + FAISS) + Local Model")
        dense_ret = DenseRetriever()
        run_approach(
            name        = "Dense (MiniLM + FAISS) + Flan-T5-base (Local)",
            retriever   = dense_ret,
            reader      = reader,
            structured  = structured,
            questions   = questions,
            output_txt  = f"{OUTPUT_DIR}/system_output_dense.txt",
            output_json = f"{OUTPUT_DIR}/run_dense_detailed.json",
        )

    print(f"""
{'='*58}
  PIPELINE COMPLETE ✓
  outputs/system_output_bm25.txt
  outputs/system_output_dense.txt
{'='*58}
→ Next: python src/05_evaluate.py
""")


if __name__ == "__main__":
    main()
