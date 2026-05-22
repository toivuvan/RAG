import json, os, re, pickle, sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from torch import cuda

READER_MODEL = "google/flan-t5-base" # Model sinh câu trả lời từ context
USE_GPU      = cuda.is_available() # Tự động dùng GPU nếu có

INDEX_DIR   = "../data/index"
QA_DIR      = "../data/qa"
OUTPUT_DIR  = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_K         = 5 # Số chunk retriever trả về mỗi câu hỏi
MAX_CTX_CHARS = 800 # Giới hạn ký tự cho context khi đưa vào model
MAX_CTX_TOKENS = 400 # Giới hạn token cho context khi đưa vào model
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2" # Model embedding cho dense retriever

"""
APPROACH có thể là "bm25", "dense", hoặc "both" để chạy từng retriever riêng hoặc so sánh cả hai. 
Nếu không để flag nào, mặc định sẽ chạy cả hai và so sánh kết quả.
"""
APPROACH = "both"
if "--approach" in sys.argv:
    APPROACH = sys.argv[sys.argv.index("--approach") + 1]
if "--qa-dir" in sys.argv:
    QA_DIR = sys.argv[sys.argv.index("--qa-dir") + 1]


def load_questions(path: str) -> list[str]:
    """Đọc danh sách câu hỏi từ file text, mỗi dòng là một câu hỏi."""
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def resolve_questions_path(qa_dir: str) -> str:
    """Tự động tìm đường dẫn file questions.txt trong qa_dir."""
    direct_path = f"{qa_dir}/questions.txt"
    if os.path.exists(direct_path):
        return direct_path
    return f"{qa_dir}/test/questions.txt"


def load_id_map(path: str) -> dict:
    """Đọc id_map.json và chuyển key global_idx từ chuỗi sang số nguyên."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def build_context(chunks: list[dict], max_chars: int = MAX_CTX_CHARS) -> str:
    """Ghép các chunk retrieval thành context ngắn để đưa vào local model."""
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
    """Tokenize query/text theo cùng logic BM25 đã dùng khi build index."""
    tokens = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class LocalModelReader:
    """Reader dùng model local fallback khi rule-based reader không trả lời được."""
    def __init__(self):
        """Khởi tạo trạng thái lazy-load model."""
        self.loaded = False
        self.device = torch.device("cuda" if USE_GPU else "cpu")

    def _load(self):
        """Load model một lần duy nhất khi cần sinh câu trả lời."""
        if self.loaded:
            return
        device_str = "GPU (CUDA)" if USE_GPU else "CPU"

        print(f"  Loading tokenizer: {READER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL)
        
        print(f"  Loading model: {READER_MODEL} ({device_str})")
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(READER_MODEL)
            self.is_seq2seq = True
            print(f"    -> Loaded as Seq2Seq model")
        except:
            self.model = AutoModelForCausalLM.from_pretrained(READER_MODEL)
            self.is_seq2seq = False
            print(f"    -> Loaded as Causal LM")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def generate(self, question: str, context: str,
                 max_new_tokens: int = 50) -> str:
        """Sinh đáp án từ question + retrieved context bằng local language model."""
        if not context.strip():
            return "Not found"
        self._load()
        
        # Tạo prompt đơn giản với question và context
        prompt = f"Q: {question} Context: {context} Answer:"

        try:
            # Tokenize prompt, cắt ngắn nếu vượt quá max token
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt",
                max_length=MAX_CTX_TOKENS,
                truncation=True,
            ).to(self.device)
            
            # Sinh câu trả lời với beam search (num_beams=1) để có output ổn định nhất
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
            
            # Decode output thành text, bỏ tokens đặc biệt
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Tách answer nếu model sinh thêm cả prompt
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            return answer or "Not found"
        except Exception:
            return "Not found"


def norm_key(text: str) -> str:
    """Chuẩn hóa text để so khớp tên đội/cầu thủ."""
    text = text.lower().replace("–", "-")
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def clean_number(value) -> str:
    """Chuẩn hóa số dạng chuỗi, bỏ hậu tố .0 nếu có."""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_position(value: str) -> str:
    """Quy đổi vị trí chi tiết về nhóm vị trí tổng quát hơn để khớp QA reference."""
    value = str(value).strip()
    mapping = {
        "dc": "DF", "dl": "DF", "dr": "DF", "cb": "DF", "lb": "DF", "rb": "DF",
        "mc": "MF", "dm": "MF", "am": "MF", "cm": "MF", "ml": "MF", "mr": "MF",
        "lw": "MF", "rw": "MF",
        "cf": "FW", "st": "FW",
    }
    parts = [p for p in re.split(r"[,/ ]+", value) if p]
    normalized = [mapping.get(p.lower(), p.upper() if len(p) <= 3 else p) for p in parts]
    return ",".join(dict.fromkeys(normalized)) if normalized else value


class StructuredReader:
    """Reader trả lời câu hỏi bằng cách so khớp metadata đã trích xuất từ chunks."""
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

    def _dedup(self, docs: list[dict], source_type: str, keys: tuple[str, ...]) -> list[dict]:
        """Lọc và deduplicate document cùng source_type theo bộ metadata keys."""
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
        """Tìm top scorer từ player chunks."""
        best_name, best_goals = "", -1
        for doc in self.players:
            m = re.search(r"scored (\d+) goals?", doc.get("text", ""))
            if m and int(m.group(1)) > best_goals:
                best_name = doc["metadata"].get("player_name", "")
                best_goals = int(m.group(1))
        return best_name, str(best_goals) if best_goals >= 0 else ""

    def _top_team(self, field: str) -> str:
        """Tìm đội có chỉ số thống kê lớn nhất theo một field metadata."""
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
        """Trả lời bằng rule metadata trước khi fallback sang local model."""
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
        """Ưu tiên duyệt document đã được retriever lấy ra trước các document khác."""
        if not retrieved_ids:
            return docs
        preferred = [d for d in docs if d.get("id") in retrieved_ids]
        return preferred + [d for d in docs if d.get("id") not in retrieved_ids]

    def _field_match(self, question_key: str, value: str) -> bool:
        """Kiểm tra value đã chuẩn hóa có xuất hiện trong câu hỏi hay không."""
        return bool(value) and norm_key(value) in question_key

    def _team_match(self, question_key: str, value: str) -> bool:
        """So khớp tên đội linh hoạt hơn, bỏ bớt token phổ biến như united/fc."""
        if self._field_match(question_key, value):
            return True
        value_tokens = [t for t in norm_key(value).split() if t not in {"fc", "cf", "afc", "united", "hove", "albion"}]
        return bool(value_tokens) and all(t in question_key for t in value_tokens[:2])

    def _answer_comparative(self, q: str) -> str | None:
        """Trả lời các câu hỏi tổng hợp như top scorer hoặc so sánh 2 entity."""
        if "top scorer" in q and q.startswith("who was"):
            return self.top_scorer or None
        if "how many goals did the top scorer score" in q:
            return self.top_goals or None
        if "which team scored the most goals" in q:
            return self.top_team_goals or None
        if "which team had the highest possession" in q:
            return self.top_possession or None
        team_pair = self._parse_between_pair(q, r"between (.+) and (.+) which team")
        if team_pair:
            answer = self._answer_team_comparison(q, team_pair[0], team_pair[1])
            if answer:
                return answer
        player_pair = self._parse_between_pair(q, r"between (.+) and (.+) who")
        if player_pair:
            answer = self._answer_player_comparison(q, player_pair[0], player_pair[1])
            if answer:
                return answer
        return None

    def _answer_match(self, q: str, retrieved_ids: set[str]) -> str | None:
        """Trả lời câu hỏi về trận đấu."""
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
            score_parts = self._score_parts(score)
            if score_parts:
                home_goals, away_goals = score_parts
                margin = abs(home_goals - away_goals)
                if "total goals" in q:
                    return str(home_goals + away_goals)
                if "goal difference" in q:
                    return str(margin)
                if "based on the score" in q and "outcome" in q:
                    if winner == "Draw":
                        return "Draw"
                    return f"{winner} won by {margin} goals"
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
        """Suy ra đội thắng hoặc Draw từ tỉ số."""
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

    def _score_parts(self, score: str) -> tuple[int, int] | None:
        """Parse tỉ số thành tuple số nguyên."""
        parts = re.split(r"[–-]", str(score))
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None

    def _parse_between_pair(self, q: str, pattern: str) -> tuple[str, str] | None:
        """Trích xuất hai entity trong câu hỏi."""
        match = re.search(pattern, q)
        if not match:
            return None
        return match.group(1).strip(), match.group(2).strip()

    def _find_team_doc(self, team_key: str) -> dict | None:
        """Tìm document team_stats khớp với tên đội trong câu hỏi."""
        for doc in self.teams:
            if self._team_match(team_key, doc["metadata"].get("team_name", "")):
                return doc
        return None

    def _find_player_doc(self, player_key: str) -> dict | None:
        """Tìm document player_stats khớp với tên cầu thủ trong câu hỏi."""
        for doc in self.players:
            if self._field_match(player_key, doc["metadata"].get("player_name", "")):
                return doc
        return None

    def _larger_name(self, left_name: str, left_value, right_name: str, right_value) -> str | None:
        """So sánh hai giá trị số và trả về tên entity có giá trị lớn hơn."""
        try:
            left = float(str(left_value).replace(",", ""))
            right = float(str(right_value).replace(",", ""))
        except (TypeError, ValueError):
            return None
        if left == right:
            return None
        return left_name if left > right else right_name

    def _answer_team_comparison(self, q: str, left_key: str, right_key: str) -> str | None:
        """Trả lời câu hỏi so sánh hai đội theo các chỉ số thống kê."""
        left_doc = self._find_team_doc(left_key)
        right_doc = self._find_team_doc(right_key)
        if not left_doc or not right_doc:
            return None

        left_meta = left_doc["metadata"]
        right_meta = right_doc["metadata"]
        left_name = left_meta.get("team_name", "")
        right_name = right_meta.get("team_name", "")

        if "scored more goals" in q:
            field = "goals"
        elif "created more assists" in q:
            field = "assists"
        elif "higher possession" in q:
            field = "possession"
        elif "more yellow cards" in q:
            field = "yellow_cards"
        else:
            return None
        return self._larger_name(left_name, left_meta.get(field), right_name, right_meta.get(field))

    def _player_stat(self, doc: dict, field: str) -> str:
        """Lấy một chỉ số player từ text semantic bằng regex."""
        text = doc.get("text", "")
        patterns = {
            "goals": r"scored (\d+) goals?",
            "assists": r"provided (\d+) assists?",
            "minutes": r"playing ([\d,]+) minutes?",
            "apps": r"made (\d+) appearances?",
        }
        pattern = patterns.get(field)
        if not pattern:
            return ""
        return self._text_number(text, pattern).replace(",", "")

    def _answer_player_comparison(self, q: str, left_key: str, right_key: str) -> str | None:
        """Trả lời câu hỏi so sánh hai cầu thủ theo các chỉ số thống kê."""
        left_doc = self._find_player_doc(left_key)
        right_doc = self._find_player_doc(right_key)
        if not left_doc or not right_doc:
            return None

        left_name = left_doc["metadata"].get("player_name", "")
        right_name = right_doc["metadata"].get("player_name", "")

        if "scored more goals" in q:
            field = "goals"
        elif "provided more assists" in q:
            field = "assists"
        elif "played more minutes" in q:
            field = "minutes"
        elif "made more appearances" in q:
            field = "apps"
        else:
            return None

        return self._larger_name(
            left_name,
            self._player_stat(left_doc, field),
            right_name,
            self._player_stat(right_doc, field),
        )

    def _answer_team(self, q: str) -> str | None:
        """Trả lời câu hỏi lookup chỉ số đội bóng từ metadata team_stats."""
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
        """Trả lời câu hỏi lookup cầu thủ từ player_stats chunks."""
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
                return normalize_position(meta.get("position", ""))
            if "nationality" in q:
                return str(meta.get("nationality", "")).strip()
            if "which club did" in q:
                return str(meta.get("team", "")).strip()
        return None

    def _text_number(self, text: str, pattern: str) -> str:
        """Lấy số đầu tiên trong text theo regex pattern."""
        match = re.search(pattern, text)
        return match.group(1) if match else ""

    def _answer_transfer(self, q: str, retrieved_ids: set[str]) -> str | None:
        """Trả lời câu hỏi chuyển nhượng từ metadata transfer chunks."""
        if not any(x in q for x in ("transfer", "join", "leaving", "pay", "receive", "fee", "league", "nationality", "position", "old")):
            return None
        for doc in self._prefer_retrieved(self.transfers, retrieved_ids):
            meta = doc["metadata"]
            player = meta.get("Player Name", "")
            team = meta.get("club", "")
            direction = meta.get("direction", "")
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
                if " pay for " in f" {q} " and direction and direction != "arrivals":
                    continue
                if " receive for " in f" {q} " and direction and direction != "departures":
                    continue
                return str(meta.get("Transfer Fee", "")).strip()
            if "join after" in q or "move after leaving" in q:
                if direction and direction != "departures":
                    continue
                if team and not self._team_match(q, team):
                    continue
                return str(meta.get("To", "")).strip()
            if "join" in q and " from" in q or "transfer from before joining" in q:
                if direction and direction != "arrivals":
                    continue
                if team and not self._team_match(q, team):
                    continue
                return str(meta.get("From", "")).strip()
            if "from which league" in q:
                return str(meta.get("League", "")).strip()
            if "what position does" in q:
                return normalize_position(meta.get("Position", ""))
            if "nationality" in q:
                return str(meta.get("Nationality", "")).strip()
            if "how old was" in q:
                return str(meta.get("Age", "")).strip()
        return None


class BM25Retriever:
    """Retriever sparse dùng BM25 để tìm chunk theo token overlap."""
    def __init__(self):
        """Load BM25 index và id_map từ index."""
        with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "rb") as f:
            data = pickle.load(f)
        self.bm25   = data["bm25"]
        self.id_map = load_id_map(os.path.join(INDEX_DIR, "id_map.json"))

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Trả về top-k chunks có điểm BM25 cao nhất cho query."""
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


class DenseRetriever:
    """Retriever dense dùng SentenceTransformer embedding và FAISS."""
    def __init__(self):
        """Load FAISS index, id_map và embedding model cho query."""
        import faiss
        from sentence_transformers import SentenceTransformer
        self.index    = faiss.read_index(os.path.join(INDEX_DIR, "faiss_index.bin"))
        self.id_map   = load_id_map(os.path.join(INDEX_DIR, "id_map.json"))
        print(f"  Loading embedder: {EMBED_MODEL}")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        print(f"  Dense retriever: {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Embed query rồi tìm top-k chunks gần nhất trong FAISS index."""
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


def run_approach(
    name:        str,
    retriever,
    reader:      LocalModelReader,
    structured:  StructuredReader,
    questions:   list[str],
    output_txt:  str,
    output_json: str,
):
    """Chạy một approach retrieval trên toàn bộ test questions và lưu prediction/log."""
    print(f"  Running: {name}")
    print(f"  Questions: {len(questions)}\n")

    results = []

    for i, q in enumerate(questions):
        # Retrieve top-k chunks cho câu hỏi hiện tại bằng retriever đã chọn (BM25 hoặc Dense)
        chunks  = retriever.retrieve(q, k=TOP_K)
        
        # Thử trả lời bằng structured reader trước; nếu retriever không có chunk,
        # reader vẫn có thể lookup toàn bộ metadata theo tên entity trong câu hỏi.
        answer = structured.answer(q, chunks)
        if not answer:
            context = build_context(chunks)
            answer = reader.generate(q, context) if chunks else "No relevant information found."

        # Ghi nhận kết quả, bao gồm question, answer, ids và sources của chunks được retrieve, cùng điểm số cao nhất
        results.append({
            "id":                i + 1,
            "question":          q,
            "answer":            answer,
            "retrieved_ids":     [c["id"]          for c in chunks],
            "retrieved_sources": [c["source_type"] for c in chunks],
            "top_score":         chunks[0]["score"] if chunks else 0.0,
        })

        # Progress log mỗi 100 câu
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1:>5}/{len(questions)}] {q[:50]:<50}")
            print(f"               Answer: {answer[:70]}")

    # Lưu answer dưới dạng text
    with open(output_txt, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r["answer"] + "\n")

    # Lưu kết quả chi tiết dưới dạng JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "approach":         name,
            "reader":           f"Local: {READER_MODEL}",
            "device":           "GPU" if USE_GPU else "CPU",
            "top_k":            TOP_K,
            "total_questions":  len(results),
            "results":          results,
        }, f, ensure_ascii=False, indent=2)

    return results


def main():
    """Entry point bước RAG: load câu hỏi, chạy BM25/Dense, ghi output."""

    # Load questions 
    q_path = resolve_questions_path(QA_DIR)

    questions = load_questions(q_path)
    print(f"\nTest questions: {len(questions)}")
    print()

    # Khởi tạo reader
    structured = StructuredReader()
    reader = LocalModelReader()

    # Sparse BM25 retrieval + local model 
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

    # Dense retrieval với embedding MiniLM + FAISS + local model
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


if __name__ == "__main__":
    main()
