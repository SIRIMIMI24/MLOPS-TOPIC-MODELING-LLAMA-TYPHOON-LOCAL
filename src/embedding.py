import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBED_BATCH_SIZE

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# Load once at module level — avoids reloading on every function call.
# First run downloads the model (~1.2GB) and caches it locally.
logging.info(f"Loading embedding model: {EMBEDDING_MODEL}")
_MODEL = SentenceTransformer(EMBEDDING_MODEL)
logging.info("Embedding model ready.")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate L2-normalized embeddings for a list of texts.

    multilingual-e5-large requires a "passage: " prefix for document embedding.
    This is a model-specific requirement — omitting it degrades embedding quality.

    Returns a numpy array of shape (len(texts), 1024).
    Embeddings are L2-normalized, ready for cosine-similarity clustering.
    """
    prefixed = [f"passage: {t}" for t in texts]
    return _MODEL.encode(
        prefixed,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,   # L2 normalize output directly
        show_progress_bar=True,
        convert_to_numpy=True
    )


def generate_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate embeddings for all messages.
    Saves numpy checkpoint every 1,000 rows to guard against crashes.
    """
    Path("../data/embeddings").mkdir(parents=True, exist_ok=True)
    formatted_dt = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    texts        = df["text_clean"].astype(str).tolist()
    n            = len(texts)

    logging.info(f"Generating embeddings for {n:,} messages...")
    all_embs = []

    for start in range(0, n, EMBED_BATCH_SIZE):
        end   = min(start + EMBED_BATCH_SIZE, n)
        batch = embed_texts(texts[start:end])
        all_embs.append(batch)

        # Checkpoint every 1,000 rows
        processed = end
        if processed % 1000 == 0 or processed == n:
            checkpoint = np.vstack(all_embs)
            np.save(
                f"../data/embeddings/checkpoint_{processed}_{formatted_dt}.npy",
                checkpoint
            )
            logging.info(f"Checkpoint: {processed:,}/{n:,} rows saved")

    all_embs_array = np.vstack(all_embs)
    np.save(f"../data/embeddings/full_{formatted_dt}.npy", all_embs_array)

    # Store as list of arrays in DataFrame for BERTopic compatibility
    df["embedding"] = list(all_embs_array)
    logging.info("Embedding complete.")
    return df