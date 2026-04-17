# ── Embedding ──────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# RTX 4050 6GB: batch size 32 is safe
# Increasing beyond 64 risks VRAM OOM during embedding
EMBED_BATCH_SIZE = 32

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "scb10x/typhoon2.1-gemma3-4b"
OLLAMA_HOST  = "http://localhost:11434"

# ── BERTopic ───────────────────────────────────────────────────────────────────
BERTOPIC_REPR_MODEL = "intfloat/multilingual-e5-large"

# ── Pipeline ───────────────────────────────────────────────────────────────────
TEXT_DELIMITER = "####"
MAX_DOC_CHARS  = 300