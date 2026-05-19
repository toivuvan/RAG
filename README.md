# EPL 2023-24 RAG Project

Hệ thống RAG hỏi đáp về Premier League 2023-24, gồm dữ liệu cầu thủ, đội bóng, trận đấu, lịch thi đấu và chuyển nhượng. Repo hiện có hai hướng retrieval:

- `BM25`: sparse retrieval bằng `rank_bm25`
- `Dense`: dense retrieval bằng SentenceTransformers + FAISS

Pipeline hiện dùng thêm structured reader trong `src/pipeline.py` để trả lời trực tiếp các câu hỏi lookup từ metadata trước khi fallback về local model.

## Cấu Trúc

```text
epl_rag_project/
├── run_all.py                       # Chạy toàn bộ chương trình
├── README.md
├── REPORT_OUTLINE.md
├── data/
│   ├── raw/                         # Raw data
│   │   └── epl_transfers/
│   ├── chunks/                      # JSONL chunks dùng để index
│   ├── index/                       # BM25, FAISS, id_map
│   └── qa/
│       ├── train/
│       ├── test/
│       └── iaa_subset.json
├── src/
│   ├── chunker.py                   # tạo chunks từ data/raw
│   ├── embedding.py                 # tạo BM25 + FAISS index
│   ├── generate_qa.py               # tạo train/test QA + IAA subset
│   ├── pipeline.py                  # chạy BM25/Dense RAG
│   └── evaluate.py                  # tính EM, F1, AnswerRecall, IAA
└── outputs/
    ├── system_output_bm25.txt
    ├── system_output_dense.txt
    ├── run_bm25_detailed.json
    ├── run_dense_detailed.json
    ├── evaluation_report.json
    └── evaluation_summary.txt
```

## Cài Đặt

Khuyến nghị dùng môi trường Conda hoặc virtualenv.

```bash
pip install pandas numpy torch transformers sentence-transformers \
            rank_bm25 faiss-cpu scikit-learn
```

Nếu dùng Apple Silicon hoặc môi trường đặc biệt, có thể cần cài `faiss-cpu` theo hướng dẫn tương ứng của hệ thống.

## Chạy Toàn Bộ

Từ thư mục gốc repo:

```bash
python run_all.py
```

Nếu chỉ cần chạy Dense hoặc BM25:

```bash
python run_all.py --approach dense
python run_all.py --approach bm25
```

## Chạy Từng Bước

```bash
# 1. Chunk raw data
python src/chunker.py

# 2. Build BM25 + FAISS indexes
python src/embedding.py

# 3. Generate QA
cd src
python generate_qa.py

# 4. Run RAG
python pipeline.py
python pipeline.py --approach bm25
python pipeline.py --approach dense

# 5. Evaluate
python evaluate.py
```

## IAA

Để tính Cohen's Kappa:

1. Mở `data/qa/iaa_subset.json`
2. Annotator 1 trả lời, mỗi dòng một đáp án, lưu vào `data/qa/iaa_annotator1.txt`
3. Annotator 2 trả lời độc lập, lưu vào `data/qa/iaa_annotator2.txt`
4. Chạy:

```bash
python run_all.py --iaa
```

## Output Chính

- `outputs/system_output_bm25.txt`: dự đoán BM25, mỗi dòng một đáp án
- `outputs/system_output_dense.txt`: dự đoán Dense, mỗi dòng một đáp án
- `outputs/run_bm25_detailed.json`: log chi tiết BM25
- `outputs/run_dense_detailed.json`: log chi tiết Dense
- `outputs/evaluation_report.json`: full metrics
- `outputs/evaluation_summary.txt`: bảng tóm tắt kết quả
