# Football RAG Local

Hệ thống RAG chạy hoàn toàn local cho dữ liệu Premier League 2023-24. Dự án được xây theo hướng rút gọn từ `Law-RAG`: chuẩn hóa corpus, build BM25, build FAISS vector bằng SentenceTransformers, truy vấn hybrid bằng Reciprocal Rank Fusion, trả lời bằng local LLM hoặc extractive fallback.

Không dùng API bên ngoài, không frontend, không Docker.

## Cấu Trúc Chính

```text
football_rag/
|- corpus.py      # Chuẩn hóa dữ liệu thành output/corpus.jsonl
|- bm25.py        # Build/query BM25 index
|- vector.py      # Build/query FAISS vector index local
|- retrieve.py    # BM25/vector/hybrid retrieval
|- answer.py      # Hỏi đáp local, có fallback không cần LLM
`- evaluate.py    # Evaluate retrieval bằng CLI

notebooks/
`- evaluate.ipynb # Evaluate có bảng và biểu đồ trực quan
```

## Cài Đặt

Chạy trong PowerShell:

```powershell
cd C:\Users\Admin\Desktop\football\RAG
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

Nếu không dùng virtual environment thì có thể chạy trực tiếp bằng Python hiện tại:

```powershell
python -m pip install -r requirements-local.txt
```

## Build Corpus

Lệnh này gom dữ liệu từ `data/processed` và transfer corpus thành một corpus chuẩn:

```powershell
python -m football_rag.corpus
```

Output:

```text
output/corpus.jsonl
```

## Build BM25

BM25 không cần tải embedding model:

```powershell
python -m football_rag.bm25 build
```

Output:

```text
output/retrieval/bm25_index.json
```

## Build Vector FAISS Local

Vector index dùng SentenceTransformers local. Lần đầu chạy có thể tải model từ Hugging Face cache về máy:

```powershell
python -m football_rag.vector build --model intfloat/multilingual-e5-small --batch-size 64
```

Output:

```text
output/retrieval/vector-local/faiss.index
output/retrieval/vector-local/vector_metadata.json
output/retrieval/vector-local/vector_manifest.json
```

Thông số index hiện tại:

```text
documents: 2694
embedding_model: intfloat/multilingual-e5-small
dimension: 384
```

## Truy Vấn Retrieval

BM25-only:

```powershell
python -m football_rag.retrieve "Which team won the Premier League 2023-24?" --mode bm25 --top-k 5
```

Vector-only:

```powershell
python -m football_rag.retrieve "How many goals did Bukayo Saka score?" --mode vector --top-k 5
```

Hybrid:

```powershell
python -m football_rag.retrieve "Manchester City transfer spending 2023 24" --mode hybrid --top-k 5
```

`hybrid` kết hợp BM25 và vector bằng Reciprocal Rank Fusion. Một số query phổ biến như câu hỏi đội vô địch được expand/rerank nhẹ để map tốt hơn với dữ liệu dạng `finished 1st`.

## Hỏi Đáp Không Cần LLM

Chạy chế độ extractive, không cần Ollama/LM Studio:

```powershell
python -m football_rag.answer "How many goals did Bukayo Saka score?" --mode hybrid --no-llm --top-k 5
```

Nếu không có local LLM, đây là chế độ ổn định nhất để kiểm tra RAG.

## Hỏi Đáp Với Local LLM

Nếu có Ollama hoặc LM Studio chạy OpenAI-compatible endpoint:

```powershell
$env:LOCAL_LLM_BASE_URL="http://127.0.0.1:11434/v1"
$env:LOCAL_CHAT_MODEL="qwen2.5:7b-instruct"
python -m football_rag.answer "Compare Arsenal and Manchester City in 2023-24" --mode hybrid --top-k 5
```

Nếu local LLM không chạy, hệ thống tự fallback sang câu trả lời extractive từ context retrieve được.

## Evaluate Bằng CLI

Chạy bộ test mặc định:

```powershell
python -m football_rag.evaluate --mode hybrid --top-k 5
```

Chạy kèm câu trả lời extractive:

```powershell
python -m football_rag.evaluate --mode hybrid --top-k 5 --answers
```

Report được ghi tại:

```text
output/evaluation/eval_report.json
```

Kết quả hiện tại trên 5 câu mẫu:

```text
hit_at_k: 1.0
mrr: 1.0
```

## Evaluate Bằng Notebook Có Biểu Đồ

Mở notebook:

```powershell
jupyter lab notebooks/evaluate.ipynb
```

Hoặc:

```powershell
jupyter notebook notebooks/evaluate.ipynb
```

Notebook sẽ:

- chạy evaluate cho `bm25`, `vector`, `hybrid`;
- hiển thị bảng kết quả từng câu hỏi;
- vẽ biểu đồ `hit@k` theo mode;
- vẽ biểu đồ `MRR` theo mode;
- vẽ biểu đồ rank của từng câu hỏi;
- vẽ biểu đồ nguồn top-1 chunk theo loại tài liệu;
- lưu report JSON vào `output/evaluation/notebook_eval_report.json`.

Nếu notebook báo thiếu `sentence_transformers` dù bạn đã chạy `pip install`, nguyên nhân thường là Jupyter đang dùng kernel Python khác với PowerShell. Trong notebook, cell đầu sẽ in ra `Notebook kernel Python`. Cài dependency vào đúng kernel đó bằng:

```powershell
"ĐƯỜNG_DẪN_PYTHON_KERNEL_IN_RA" -m pip install -r requirements-local.txt
```

Ví dụ nếu notebook in:

```text
Notebook kernel Python: C:\Users\Admin\Desktop\football\RAG\.venv\Scripts\python.exe
```

thì chạy:

```powershell
C:\Users\Admin\Desktop\football\RAG\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
```

Notebook cũng có cell tự kiểm tra và cài package thiếu vào đúng kernel đang chạy.

Khuyến nghị trong VS Code/Jupyter: chọn kernel `football_rag (local)` ở góc phải trên notebook. Kernel này đã được đăng ký riêng cho dự án football RAG.

## Tạo Bộ Evaluate Riêng

Tạo file JSON dạng list:

```json
[
  {
    "id": "saka_goals",
    "question": "How many goals did Bukayo Saka score?",
    "expected_contains": ["Bukayo Saka", "16 goals"]
  },
  {
    "id": "league_winner",
    "question": "Which team won the Premier League 2023-24?",
    "expected_contains": ["Manchester City", "finished 1st"]
  }
]
```

Chạy:

```powershell
python -m football_rag.evaluate --cases path\to\cases.json --mode hybrid --top-k 5
```

## Quy Trình Chạy Từ Đầu

```powershell
cd C:\Users\Admin\Desktop\football\RAG
python -m football_rag.corpus
python -m football_rag.bm25 build
python -m football_rag.vector build --model intfloat/multilingual-e5-small --batch-size 64
python -m football_rag.evaluate --mode hybrid --top-k 5
python -m football_rag.answer "Which team won the Premier League 2023-24?" --mode hybrid --no-llm
```

## Ghi Chú

- `output/` là artifact build local, đã được ignore khỏi Git.
- `vector` và `hybrid` cần `faiss-cpu` và `sentence-transformers`.
- Lần đầu dùng SentenceTransformers có thể chậm vì tải model.
- Một số dữ liệu nguồn có lỗi encoding sẵn, hệ thống vẫn index và retrieve được nhưng text hiển thị có thể có ký tự lỗi ở một vài chunk.
