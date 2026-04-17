import pandas as pd
import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)


def load_chat_data(
    file_path: str,
    message_col: str,
    role_col: str = None,
    customer_value: str = None,
    session_col: str = None,
) -> pd.DataFrame:
    """
    Load customer chat data from CSV or Excel.
    Filters to customer-only messages when role column is available.

    Supported formats
    -----------------
    Format A — with sender column:
    | session_id | sender   | message                      |
    |------------|----------|------------------------------|
    | 001        | customer | อยากรู้ค่าส่งไปเชียงใหม่    |
    | 001        | agent    | สวัสดีครับ ยินดีช่วยเหลือ   |

    Format B — customer messages only (no role column):
    | session_id | message                      |
    |------------|------------------------------|
    | 001        | อยากรู้ค่าส่งไปเชียงใหม่    |
    """
    logging.info(f"Loading: {file_path}")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    elif file_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported format. Use .csv, .xlsx, or .xls")

    logging.info(f"Loaded {len(df):,} rows | Columns: {list(df.columns)}")

    # Filter customer messages only
    if role_col and customer_value:
        before = len(df)
        df = df[df[role_col] == customer_value].copy()
        logging.info(f"Customer filter: {before:,} → {len(df):,} rows")

    # Drop empty messages
    df = df.dropna(subset=[message_col])
    df = df[df[message_col].astype(str).str.strip() != ""]

    # Standardize column names
    df = df.rename(columns={message_col: "raw_message"})
    if session_col:
        df = df.rename(columns={session_col: "session_id"})

    df = df.reset_index(drop=True)
    logging.info(f"Final: {len(df):,} customer messages ready")
    return df


def preview(df: pd.DataFrame, n: int = 5) -> None:
    """Quick sanity check before processing."""
    print("\n=== Data Preview ===")
    print(df[["raw_message"]].head(n).to_string())
    print(f"\nTotal messages : {len(df):,}")
    print(f"Avg length     : {df['raw_message'].str.len().mean():.0f} chars")
    print(f"Max length     : {df['raw_message'].str.len().max()} chars")