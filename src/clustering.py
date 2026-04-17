import numpy as np
import pandas as pd
import joblib
import logging
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from pathlib import Path
from datetime import datetime
from config import BERTOPIC_REPR_MODEL

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)


def run_bertopic(df: pd.DataFrame) -> tuple[pd.DataFrame, BERTopic, dict]:
    """
    Cluster customer messages into topics using BERTopic.

    BERTopic pipeline:
    ┌─────────────────────────────────────────────────────┐
    │ Pre-computed Typhoon/E5 Embeddings (1024-dim)       │
    │          ↓                                          │
    │ UMAP — reduce to lower dimensions before clustering │
    │          ↓                                          │
    │ HDBSCAN — density-based clustering                  │
    │          ↓                                          │
    │ c-TF-IDF — extract representative keywords         │
    └─────────────────────────────────────────────────────┘

    Pre-computed embeddings are passed directly to BERTopic.
    The sentence_model parameter is used only for BERTopic's
    internal topic representation — not for re-embedding.
    """
    Path("../models").mkdir(parents=True, exist_ok=True)
    Path("../data/topics").mkdir(parents=True, exist_ok=True)
    formatted_dt = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")

    # Stack embeddings — already L2-normalized from embedding.py
    emb_array = np.array(df["embedding"].tolist())
    docs      = df["text_clean"].astype(str).tolist()

    # Reuse same model — no extra download required
    sentence_model = SentenceTransformer(BERTOPIC_REPR_MODEL)

    # Scale min_topic_size with corpus size:
    # Too small → noisy micro-topics
    # Too large → merges genuinely distinct customer intents
    n        = len(df)
    min_size = 5 if n < 1000 else (10 if n < 5000 else 20)
    logging.info(f"Fitting BERTopic | {n:,} messages | min_topic_size={min_size}")

    model = BERTopic(
        embedding_model=sentence_model,
        language="multilingual",      # Required for Thai-English mixed input
        calculate_probabilities=True,
        min_topic_size=min_size,
        nr_topics="auto",             # Auto-merge semantically similar topics
        verbose=True
    )

    topics, probs    = model.fit_transform(docs, emb_array)
    df["topic"]      = topics
    # probs is 2-D (n_docs, n_topics) when calculate_probabilities=True;
    # store only the max probability across topics for each document.
    if probs is not None and np.ndim(probs) == 2:
        df["topic_prob"] = probs.max(axis=1)
    else:
        df["topic_prob"] = probs

    # Reduce outliers: reassign topic -1 documents only if any exist
    if -1 in topics:
        new_topics = model.reduce_outliers(docs, topics, strategy="distributions")
    else:
        new_topics = topics
    df["topic_final"] = new_topics

    # Persist model for future inference on new messages
    joblib.dump(model, f"../models/bertopic_{formatted_dt}.joblib")
    df.to_csv(
        f"../data/topics/raw_clusters_{formatted_dt}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    rep_docs = model.get_representative_docs()
    n_topics = len(set(new_topics)) - 1  # Exclude -1 outlier cluster
    logging.info(f"Topics identified: {n_topics}")
    return df, model, rep_docs