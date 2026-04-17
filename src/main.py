"""
Thai Customer Chat Topic Modeling
===================================
Provider  : Typhoon (SCB10X) — fully local via Ollama
Embedding : intfloat/multilingual-e5-large (local, sentence-transformers)
LLM Label : typhoon2.1-gemma3-4b (local, Ollama)
No API keys. No internet required after setup. No usage cost.

Prerequisites
-------------
1. Install Ollama       : https://ollama.com/download
2. Pull model           : ollama pull typhoon2.1-gemma3-4b
3. Install dependencies : pip install -r requirements.txt

Usage
-----
    cd src
    python main.py
"""

import pandas as pd
from pathlib import Path
from data_loader   import load_chat_data, preview
from preprocessing import preprocess_dataframe
from embedding     import generate_embeddings
from clustering    import run_bertopic
from labeling      import label_topics
from reporting     import build_summary, plot_top_topics, export


# ╔══════════════════════════════════════════════════════╗
# ║           USER CONFIGURATION — edit here only       ║
# ╠══════════════════════════════════════════════════════╣
FILE_PATH      = "../data/raw/chat_data.csv"  # .csv or .xlsx
MESSAGE_COL    = "message"     # Column containing message text
ROLE_COL       = None     # Sender column — set None if not available
CUSTOMER_VALUE = None    # Value identifying customer messages
SESSION_COL    = "session_id"  # Session column — set None if not available
TOP_N_CHART    = 15            # Number of topics to show in bar chart
# ╚══════════════════════════════════════════════════════╝


def run():
    # Create all required directories upfront
    for d in ["../data/processed", "../data/embeddings",
              "../data/topics", "../models", "../outputs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 60)
    print("  Thai Customer Chat Topic Modeling")
    print("  Local Stack: E5-Large + Typhoon Gemma3-4B via Ollama")
    print("═" * 60)

    # ── Step 1: Load ───────────────────────────────────────────
    print("\n[1/6] Loading chat data...")
    df = load_chat_data(
        file_path=FILE_PATH,
        message_col=MESSAGE_COL,
        role_col=ROLE_COL,
        customer_value=CUSTOMER_VALUE,
        session_col=SESSION_COL
    )
    preview(df)

    # ── Step 2: Preprocess ─────────────────────────────────────
    print("\n[2/6] Preprocessing (Thai-English mixed text)...")
    df = preprocess_dataframe(df)
    df.to_csv(
        "../data/processed/messages_clean.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ── Step 3: Embed ──────────────────────────────────────────
    print("\n[3/6] Generating local embeddings (multilingual-e5-large)...")
    df = generate_embeddings(df)

    # ── Step 4: Cluster ────────────────────────────────────────
    print("\n[4/6] Clustering with BERTopic...")
    df, bertopic_model, rep_docs = run_bertopic(df)

    # ── Step 5: Label ──────────────────────────────────────────
    print("\n[5/6] Labeling topics with Typhoon (local Ollama)...")
    topic_labels = label_topics(rep_docs)

    # Merge labels back into the main DataFrame
    df["topic_name"] = df["topic_final"].map(
        {int(k): v.get("topic_name", "N/A") for k, v in topic_labels.items()}
    )
    df["topic_description"] = df["topic_final"].map(
        {int(k): v.get("topic_description", "") for k, v in topic_labels.items()}
    )

    # ── Step 6: Report ─────────────────────────────────────────
    print("\n[6/6] Building report...")
    summary = build_summary(df)
    export(df, summary)
    plot_top_topics(summary, top_n=TOP_N_CHART)

    # Print console summary
    print("\n" + "═" * 60)
    print("  Top Customer Topics by Volume")
    print("═" * 60)
    print(
        summary[["rank", "topic_name", "message_count", "percentage"]]
        .to_string(index=False)
    )
    print(f"\nTopics identified : {len(summary)}")
    print(f"Messages analyzed : {len(df):,}")
    print(f"Outputs saved to  : ../outputs/")

    return df, summary


if __name__ == "__main__":
    df, summary = run()