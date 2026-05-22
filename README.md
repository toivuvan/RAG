# EPL 2023-24 RAG Project

Dự án xây dựng hệ thống Retrieval-Augmented Generation (RAG) để hỏi đáp về Premier League mùa 2023-24. Dữ liệu gồm thông tin cầu thủ, đội bóng, trận đấu, lịch thi đấu và chuyển nhượng.

Hệ thống hiện hỗ trợ hai hướng retrieval:

- `BM25`: sparse retrieval bằng `rank_bm25`
- `Dense`: dense retrieval bằng SentenceTransformers + FAISS

Trong `src/pipeline.py`, hệ thống dùng thêm `StructuredReader` để trả lời trực tiếp các câu hỏi có thể đọc từ metadata trước khi fallback sang local model `google/flan-t5-base`.

## Cấu Trúc Thư Mục Dự Án

```text
epl_rag_project/
├── README.md
├── .gitignore
├── data/
│   ├── raw/                          # Dữ liệu gốc
│   │   ├── epl_transfers/           
│   │   ├── standard_stats.csv
│   │   ├── shooting_stats.csv
│   │   ├── playing_time_stats.csv
│   │   ├── misc_stats.csv
│   │   ├── keeper_stats.csv
│   │   ├── team_stats.csv
│   │   ├── team_shooting_stats.csv
│   │   ├── team_playing_time_stats.csv
│   │   ├── team_misc_stats.csv
│   │   ├── schedule.csv
│   │   ├── standings.csv
│   │   ├── player_metadata.csv
│   │   └── metadata_clubs.json
│   ├── chunks/                       # Các file chunks
│   │   ├── all_players.jsonl
│   │   ├── all_teams.jsonl
│   │   ├── all_matches.jsonl
│   │   ├── transfers_chunks.jsonl
│   │   ├── schedule_chunks.jsonl
│   │   └── corpus_unified.jsonl 
│   ├── index/                        # Index phục vụ cho retrieval
│   │   ├── bm25_corpus.pkl
│   │   ├── faiss_index.bin
│   │   └── id_map.json
│   └── qa/                           # Bộ câu hỏi sinh tự động
│       ├── train/
│       │   ├── qa_pairs.json
│       │   ├── questions.txt         # Câu hỏi
│       │   └── reference_answers.txt # Câu trả lời chuẩn
│       ├── test/
│       │   ├── qa_pairs.json
│       │   ├── questions.txt
│       │   └── reference_answers.txt
│       ├── iaa_subset.json          # Tập câu hỏi mẫu để tính Inter-Annotator Agreement
│       └── split_metadata.json      # Thống kê/cấu hình chia train-test theo entity
├── src/
│   ├── run_all.py                   # Chạy toàn bộ pipeline
│   ├── chunker.py                   # Tạo chunks từ dữ liệu gốc
│   ├── embedding.py                 # Tạo corpus_unified, BM25, FAISS, id_map
│   ├── generate_qa.py               # Tạo QA train/test theo entity split
│   ├── pipeline.py                  # Chạy BM25/Dense retrieval + reader
│   └── evaluate.py                  # Tính EM, F1, AnswerRecall, IAA
└── outputs/
    ├── system_output_bm25.txt       # Output của BM25
    ├── system_output_dense.txt      # Output của Dense
    ├── run_bm25_detailed.json       # Log chi tiết từng câu hỏi của BM25
    ├── run_dense_detailed.json      # Log chi tiết từng câu hỏi của Dense
    ├── evaluation_report.json       # Báo cáo đánh giá đầy đủ dạng JSON
    └── evaluation_summary.txt       # Tóm tắt kết quả đánh giá dạng text
```

## Cài Đặt

Khuyến nghị dùng Conda hoặc virtualenv.

```bash
pip install pandas numpy torch transformers sentence-transformers rank_bm25 faiss-cpu scikit-learn
```

Nếu dùng Apple Silicon hoặc môi trường đặc biệt, `faiss-cpu` có thể cần cài theo hướng dẫn riêng của hệ thống.

## Chạy Toàn Bộ Pipeline

Từ thư mục gốc của project:

```bash
python src/run_all.py
```

Lệnh trên chạy theo thứ tự:

```text
chunker.py -> embedding.py -> generate_qa.py -> pipeline.py -> evaluate.py
```

Chỉ chạy một retrieval approach:

```bash
python src/run_all.py --approach bm25
python src/run_all.py --approach dense
```

## Chạy Từng Bước

```bash
cd src

python chunker.py
python embedding.py
python generate_qa.py
python pipeline.py
python evaluate.py
```

Chạy riêng pipeline với từng approach:

```bash
python pipeline.py --approach bm25
python pipeline.py --approach dense
python pipeline.py --approach both
```

## Ý Nghĩa Từng Bước

`chunker.py` đọc dữ liệu thô trong `data/raw/` và tạo các file JSONL trong `data/chunks/`.

`embedding.py` đọc các chunk, chuẩn hóa về `corpus_unified.jsonl`, sau đó tạo:

- `data/index/bm25_corpus.pkl`
- `data/index/faiss_index.bin`
- `data/index/id_map.json`

`generate_qa.py` sinh bộ câu hỏi train/test vào `data/qa/`. File này dùng entity split để giảm leakage giữa train và test, đồng thời tạo `iaa_subset.json` và `split_metadata.json`.

`data/qa/iaa_subset.json` là tập con câu hỏi dùng cho IAA. Hai annotator trả lời độc lập các câu hỏi trong file này, sau đó `evaluate.py --iaa` dùng hai file đáp án `iaa_annotator1.txt` và `iaa_annotator2.txt` để tính Cohen's Kappa.

`data/qa/split_metadata.json` lưu thông tin về quá trình chia train/test theo entity, ví dụ số câu hỏi, số entity trong từng split và mức độ overlap. File này giúp kiểm tra train/test có được chia đúng và hạn chế leakage giữa hai tập hay không.

`pipeline.py` chạy retrieval bằng BM25 hoặc Dense, sau đó dùng `StructuredReader` và local model để sinh câu trả lời. Kết quả được lưu vào `outputs/`.

`evaluate.py` so sánh prediction với reference answers và tính các metric:

- Exact Match (EM)
- Token-level F1
- Answer Recall
- F1 theo question type
- F1 theo difficulty
- Cohen's Kappa nếu chạy với `--iaa`

## IAA

Để tính Inter-Annotator Agreement (IAA):

1. Mở `data/qa/iaa_subset.json`
2. Annotator 1 trả lời từng câu, mỗi dòng một đáp án, lưu vào:

```text
data/qa/iaa_annotator1.txt
```

3. Annotator 2 trả lời độc lập, mỗi dòng một đáp án, lưu vào:

```text
data/qa/iaa_annotator2.txt
```

4. Chạy:

```bash
python evaluate.py --iaa
```

## Output Chính

- `outputs/system_output_bm25.txt`: câu trả lời của hệ BM25, mỗi dòng ứng với một câu hỏi test
- `outputs/system_output_dense.txt`: câu trả lời của hệ Dense, mỗi dòng ứng với một câu hỏi test
- `outputs/run_bm25_detailed.json`: log chi tiết từng câu hỏi khi chạy BM25, gồm question, answer, retrieved chunk ids, source type và top retrieval score.
- `outputs/run_dense_detailed.json`: log chi tiết từng câu hỏi khi chạy Dense, gồm question, answer, retrieved chunk ids, source type và top retrieval score.
- `outputs/evaluation_report.json`: báo cáo đánh giá đầy đủ dạng JSON, gồm overall metrics, kết quả theo question type, difficulty và kết quả từng câu hỏi.
- `outputs/evaluation_summary.txt`: bản tóm tắt dễ đọc của kết quả đánh giá, dùng để xem nhanh EM, F1, AnswerRecall và F1 theo từng nhóm câu hỏi.
