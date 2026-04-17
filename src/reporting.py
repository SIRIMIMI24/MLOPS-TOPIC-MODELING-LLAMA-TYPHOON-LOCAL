import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate topic counts and percentage share.
    Excludes unidentifiable topics from the business-facing report.
    """
    summary = (
        df[df["topic_name"].notna()]
        [df["topic_name"] != "ไม่สามารถระบุหัวข้อได้"]
        .groupby(["topic_name", "topic_description"])
        .size()
        .reset_index(name="message_count")
        .sort_values("message_count", ascending=False)
        .reset_index(drop=True)
    )
    summary["percentage"] = (summary["message_count"] / len(df) * 100).round(1)
    summary.insert(0, "rank", summary.index + 1)
    return summary


def _load_thai_font():
    """
    Attempt to load a Thai-compatible system font for matplotlib.
    Falls back silently to system default if none found.
    """
    candidates = ["Sarabun", "TH Sarabun New", "Noto Sans Thai", "Tahoma", "Arial"]
    for name in candidates:
        matches = [f for f in fm.findSystemFonts() if name.lower() in f.lower()]
        if matches:
            return fm.FontProperties(fname=matches[0])
    return None


def plot_top_topics(summary: pd.DataFrame, top_n: int = 15) -> None:
    """
    Horizontal bar chart showing the top N customer topics by message volume.
    Each bar is annotated with count and percentage share.
    """
    Path("../outputs").mkdir(parents=True, exist_ok=True)
    thai_font = _load_thai_font()

    plot_df = summary.head(top_n).sort_values("message_count", ascending=True)
    max_val = plot_df["message_count"].max()

    fig, ax = plt.subplots(figsize=(13, max(6, len(plot_df) * 0.55)))

    bars = ax.barh(
        plot_df["topic_name"],
        plot_df["message_count"],
        color="#1c5872",
        edgecolor="white",
        linewidth=0.5
    )

    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        ax.text(
            bar.get_width() + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{int(row["message_count"]):,}  ({row["percentage"]}%)',
            va="center",
            fontsize=9,
            color="#333333"
        )

    font_kw = {"fontproperties": thai_font} if thai_font else {}
    ax.set_xlabel("จำนวน Messages", **font_kw)
    ax.set_title(
        "หัวข้อที่ลูกค้าถามมากที่สุด",
        fontsize=14, pad=15, **font_kw
    )

    if thai_font:
        for label in ax.get_yticklabels():
            label.set_fontproperties(thai_font)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max_val * 1.18)
    plt.tight_layout()
    plt.savefig("../outputs/top_topics.png", dpi=300, bbox_inches="tight")
    logging.info("Chart saved: ../outputs/top_topics.png")
    plt.show()


def export(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Export full labeled dataset and summary report as CSV.
    utf-8-sig encoding ensures Thai characters render correctly in Excel.
    """
    Path("../outputs").mkdir(parents=True, exist_ok=True)

    df.to_csv(
        "../outputs/messages_labeled.csv",
        index=False,
        encoding="utf-8-sig"
    )
    summary.to_csv(
        "../outputs/topic_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )
    logging.info("Exported: messages_labeled.csv, topic_summary.csv")