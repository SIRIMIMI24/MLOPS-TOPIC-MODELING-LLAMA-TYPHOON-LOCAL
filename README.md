# Thai Customer Chat — Topic Modeling

A system for analyzing chat topics from Thai customer messages, **100% local**, without requiring an API key and at no cost.

**Stack:** `multilingual-e5-large` (embedding) · `BERTopic` (clustering) · `Typhoon via Ollama` (labeling)

---

## Pipeline

```
Chat CSV/Excel
     │
     ▼
[1] Load & Filter         — กรองเฉพาะ message ลูกค้า
     │
     ▼
[2] Preprocess            — Normalize · Tokenize (PyThaiNLP) · Remove stop words
     │
     ▼
[3] Embed                 — intfloat/multilingual-e5-large (1024-dim, L2-normalized)
     │
     ▼
[4] BERTopic Cluster      — UMAP → HDBSCAN → c-TF-IDF
     │
     ▼
[5] Label with Typhoon    — scb10x/typhoon2.1-gemma3-4b via Ollama (local LLM)
     │
     ▼
[6] Report & Export       — CSV + Bar chart PNG
```

---

## Project Structure

```
├── src/
│   ├── main.py            # Entry point — runs full pipeline
│   ├── config.py          # Model names, batch sizes, paths
│   ├── data_loader.py     # Load CSV/Excel, filter customer messages
│   ├── preprocessing.py   # Thai-English normalize, tokenize, stop words
│   ├── embedding.py       # multilingual-e5-large embeddings
│   ├── clustering.py      # BERTopic fit + outlier reduction
│   ├── labeling.py        # Typhoon LLM topic labeling via Ollama
│   └── reporting.py       # Summary CSV + bar chart
│
├── notebook/
│   └── topic_modeling_colab.ipynb   # Google Colab version (self-contained)
│
├── data/
│   ├── raw/               # Input: chat_data.csv
│   ├── processed/         # messages_clean.csv
│   ├── embeddings/        # .npy checkpoints
│   └── topics/            # raw_clusters.csv, labels.json
│
├── models/                # Saved BERTopic .joblib
├── outputs/               # messages_labeled.csv, topic_summary.csv, top_topics.png
└── requirements.txt
```

---

## Quick Start (Local)

### Prerequisites

```bash
# 1. Install Ollama
# Windows/Mac: https://ollama.com/download
# Linux:
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull Typhoon model (~2.6 GB)
ollama pull scb10x/typhoon2.1-gemma3-4b
```

### Install & Run

```bash
# Clone repo
git clone <repo-url>
cd MLOPS-TOPIC-MODELING-LLAMA-TYPHOON-LOCAL

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -e .

# Run pipeline
cd src
python main.py
```

### Output Files

| File | Description |
|---|---|
| `outputs/messages_labeled.csv` | ข้อความทุก message พร้อม topic |
| `outputs/topic_summary.csv` | สรุป topic ทั้งหมด + จำนวน + % |
| `outputs/top_topics.png` | Bar chart top topics |
| `data/topics/labels_*.json` | Raw label output จาก LLM |
| `models/bertopic_*.joblib` | Saved BERTopic model |

---

## Run on Google Colab

เปิดไฟล์ `notebook/topic_modeling_colab.ipynb` บน [Google Colab](https://colab.research.google.com)

> **Runtime → Change runtime type → T4 GPU** ก่อน Run

Notebook จะติดตั้ง Ollama, pull model, และ run pipeline ครบทุก step ในที่เดียว

---

## Configuration

แก้ค่าใน `src/config.py` หรือ section **USER CONFIGURATION** ใน `main.py`

```python
# main.py
FILE_PATH      = "../data/raw/chat_data.csv"
MESSAGE_COL    = "message"      # ชื่อ column ข้อความ
ROLE_COL       = None           # ชื่อ column sender (None ถ้าไม่มี)
CUSTOMER_VALUE = None           # ค่าที่แทน customer (None ถ้าไม่มี)
SESSION_COL    = "session_id"   # ชื่อ column session (None ถ้าไม่มี)
TOP_N_CHART    = 15             # จำนวน topics ในกราฟ
```

```python
# config.py
OLLAMA_MODEL     = "scb10x/typhoon2.1-gemma3-4b"   # หรือ scb10x/typhoon2.5-qwen3-4b
EMBEDDING_MODEL  = "intfloat/multilingual-e5-large"
EMBED_BATCH_SIZE = 32   # เพิ่มเป็น 64 ถ้า VRAM > 8 GB
```

### Input Data Format

**Format A** — มี sender column:

| session_id | sender | message |
|---|---|---|
| 001 | customer | อยากรู้ค่าส่งไปเชียงใหม่ |
| 001 | agent | สวัสดีครับ ยินดีช่วยเหลือ |

**Format B** — เฉพาะ message ลูกค้า (ไม่มี role column):

| session_id | message |
|---|---|
| 001 | อยากรู้ค่าส่งไปเชียงใหม่ |

---

## Requirements

| Package | Purpose |
|---|---|
| `bertopic` | Topic modeling |
| `sentence-transformers` | multilingual-e5-large embeddings |
| `pythainlp` | Thai tokenization & normalization |
| `ollama` | Local LLM inference (Typhoon) |
| `umap-learn` / `hdbscan` | Dimensionality reduction & clustering |
| `torch` | Model inference backend |

---

## Why Local?

- **ไม่มี API Key** — ไม่ต้องสมัคร ไม่มีค่าใช้จ่าย
- **ความปลอดภัยของข้อมูล** — ข้อความลูกค้าไม่ออกจากเครื่อง
- **Reproducible** — ผล deterministic (`temperature=0`)
- **Thai-first** — Typhoon ออกแบบมาสำหรับภาษาไทยโดยเฉพาะ
