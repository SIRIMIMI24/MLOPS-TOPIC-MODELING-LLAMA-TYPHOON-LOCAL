import re
import logging
import pandas as pd
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize as thai_normalize

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# ── Stop Words ─────────────────────────────────────────────────────────────────
THAI_STOPS = set(thai_stopwords())

# Polite particles and filler words common in Thai customer chat.
# These appear in virtually every message but carry zero topic signal.
CHAT_STOPS = {
    "ครับ", "ค่ะ", "คะ", "นะ", "นะครับ", "นะคะ", "ด้วยนะ",
    "ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ", "สวัสดี", "สวัสดีครับ",
    "โอเค", "โอเค", "ok", "okay",
    "อยาก", "ต้องการ", "ขอ", "ได้", "มี", "ไม่",
    "ถาม", "สอบถาม", "รบกวน", "ช่วย", "หน่อย",
}

ALL_THAI_STOPS = THAI_STOPS.union(CHAT_STOPS)

# Minimal English stop words — avoids nltk dependency
ENGLISH_STOPS = {
    "i", "me", "my", "we", "you", "he", "she", "it", "they",
    "is", "am", "are", "was", "were", "be", "been", "have", "has",
    "do", "does", "did", "will", "would", "could", "should",
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "hi", "hello", "thanks", "thank",
    "please", "yes", "no", "ok", "okay",
}


def detect_lang(text: str) -> str:
    """
    Classify message language as 'th', 'en', or 'mixed'.
    Thai-English code-switching is extremely common in Thai customer chat.
    """
    thai = len(re.findall(r'[\u0E00-\u0E7F]', text))
    eng  = len(re.findall(r'[a-zA-Z]', text))
    tot  = thai + eng
    if tot == 0:
        return "other"
    r = thai / tot
    if r >= 0.8:
        return "th"
    elif r <= 0.2:
        return "en"
    return "mixed"


def normalize(text: str) -> str:
    """
    Normalize Thai-English mixed text:
    1. Fix Thai Unicode character ordering and vowel positions
    2. Remove URLs, phone numbers, emails (noise + privacy)
    3. Remove emojis and special characters
    4. Remove standalone numbers (order IDs carry no topic signal)
    5. Lowercase English only (Thai has no case)
    6. Strip non-Thai, non-English characters
    7. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = thai_normalize(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\b0\d{8,9}\b", " ", text)           # Thai phone numbers
    text = re.sub(r"[\w.-]+@[\w.-]+\.\w+", " ", text)   # Emails
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text, flags=re.UNICODE)  # Emojis
    text = re.sub(r"[\U0001F600-\U0001F64F]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\b\d+\b", " ", text)                 # Standalone numbers
    text = re.sub(r'[a-zA-Z]+', lambda m: m.group().lower(), text)  # Lowercase EN only
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z\s]", " ", text)           # Keep TH + EN only
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenize Thai-English mixed text using language-appropriate strategies:
    - Thai segments → PyThaiNLP newmm (dictionary-based, fast)
    - English segments → whitespace split
    """
    if not text:
        return []

    tokens = []
    for seg in re.split(r'(\s+)', text):
        seg = seg.strip()
        if not seg:
            continue
        if re.search(r'[\u0E00-\u0E7F]', seg):
            tokens.extend(word_tokenize(seg, engine="newmm", keep_whitespace=False))
        elif re.search(r'[a-zA-Z]', seg):
            tokens.extend(seg.split())
    return tokens


def remove_stops(tokens: list[str]) -> list[str]:
    """Remove Thai and English stop words. Discard tokens < 2 characters."""
    return [
        t for t in tokens
        if t not in ALL_THAI_STOPS
        and t not in ENGLISH_STOPS
        and len(t) >= 2
    ]


def preprocess_one(text: str) -> str:
    """Full single-message pipeline: normalize → tokenize → remove stops → rejoin."""
    return " ".join(remove_stops(tokenize(normalize(text))))


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the entire DataFrame.

    New columns added:
    - text_clean  : Preprocessed text used as embedding input
    - lang_type   : Detected language ('th' / 'en' / 'mixed')
    - char_count  : Raw character count (for outlier inspection)
    """
    logging.info("Detecting language per message...")
    df["lang_type"] = df["raw_message"].apply(detect_lang)
    logging.info(f"Language distribution:\n{df['lang_type'].value_counts().to_string()}")

    logging.info("Preprocessing messages...")
    df["text_clean"] = df["raw_message"].apply(preprocess_one)
    df["char_count"] = df["raw_message"].str.len()

    # Drop messages with insufficient content after cleaning
    before = len(df)
    df = df[df["text_clean"].str.len() >= 5].reset_index(drop=True)
    logging.info(f"Removed short messages: {before:,} → {len(df):,} rows")
    return df