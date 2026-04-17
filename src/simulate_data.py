"""
Simulation Script — Generate Sample Chat Data
==============================================
Creates a synthetic chat_data.csv with two columns only:
  - message    (MESSAGE_COL)
  - session_id (SESSION_COL)

Output: ../artifacts/raw/chat_data.csv

Run:
    python simulate_data.py
"""

import random
import pandas as pd
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────
random.seed(42)

# ── Sample customer messages (Thai-English mixed) ─────────────
MESSAGES = [
    # Shipping / delivery
    "สั่งของไปแล้ว 3 วัน ยังไม่ได้รับเลยค่ะ",
    "ของยังไม่มาเลย tracking ไม่อัปเดตเลย",
    "เมื่อไหร่ของจะถึงคะ order เลขที่ 10293",
    "ส่งของไปแล้วแต่คนรับบอกไม่ได้รับ",
    "delivery รอบนี้ช้ามากเลยครับ",
    "พัสดุหายไปเลยค่ะ ช่วยตามให้หน่อยได้ไหม",
    "ของถูกส่งผิดที่อยู่ครับ",
    "ต้องการเปลี่ยนที่อยู่จัดส่งหลังจากสั่งแล้ว",
    "ขนส่งบอกว่าไม่มีคนรับแต่ฉันอยู่บ้านตลอด",
    "กล่องพัสดุบุบมาเลยค่ะ สินค้าข้างในเสียหาย",

    # Payment / billing
    "จ่ายเงินไปแล้วแต่ order ยังไม่ยืนยัน",
    "โอนเงินซ้ำสองครั้งขอ refund ได้ไหม",
    "บัตรเครดิตถูกตัดเงินแต่ order ไม่สำเร็จ",
    "ต้องการใบกำกับภาษีครับ",
    "ชำระด้วย QR แล้วแต่ระบบบอก payment failed",
    "อยากผ่อน 0% ได้ไหมคะ",
    "ยอดเงินที่หักไปไม่ตรงกับราคาสินค้า",
    "ขอ refund เงินคืนหน่อยได้ไหมครับ",
    "coupon ที่ใช้ไม่ได้รับส่วนลด",
    "มีค่าจัดส่งเพิ่มทำไมตอนที่ checkout",

    # Product / quality
    "สินค้าที่ได้รับไม่ตรงกับรูปในเว็บเลย",
    "ของปลอมแน่ๆ เลยค่ะ ไม่ได้ของแท้",
    "สีไม่ตรงกับที่สั่งเลยครับ",
    "size ไม่ตรงตามที่ระบุ ใหญ่กว่าจริง",
    "สินค้ามีรอยขีดข่วนมาตั้งแต่แกะกล่อง",
    "ของใช้ได้แค่วันเดียวก็พัง",
    "ได้รับสินค้าหมดอายุมาเลยค่ะ",
    "อยากรู้วัสดุที่ใช้ผลิตสินค้านี้ครับ",
    "สินค้าชิ้นนี้ washable ไหมคะ",
    "มี manual ภาษาไทยไหมครับ",

    # Return / exchange
    "อยากคืนสินค้าทำได้ไหมครับ",
    "เปลี่ยนไซส์ได้ไหมคะ ได้รับผิดมา",
    "ส่งของผิด ขอเปลี่ยนให้ถูกต้องด้วย",
    "return policy ของร้านเป็นยังไงบ้าง",
    "ส่งคืนแล้วแต่ยังไม่ได้รับเงินคืนเลย",
    "อยากเปลี่ยนสินค้าเป็นรุ่นใหม่กว่า",
    "ของชำรุดตั้งแต่แกะกล่อง ขอเปลี่ยนได้ไหม",

    # Account / login
    "login ไม่ได้เลยค่ะ ลืม password",
    "OTP ไม่มาที่มือถือ",
    "อยากเปลี่ยนเบอร์โทรที่ผูกกับบัญชี",
    "บัญชีถูก suspend ทำไมคะ",
    "ลบบัญชีออกได้ไหมครับ",
    "อยากเปลี่ยน email ของบัญชี",

    # Promotion / discount
    "โค้ดส่วนลดที่ได้รับทาง SMS ใช้ไม่ได้",
    "flash sale ราคาที่โฆษณาไม่ตรงกับตอน checkout",
    "สมาชิก VIP ได้ส่วนลดเพิ่มไหมคะ",
    "มีโปรอะไรบ้างช่วงเดือนนี้",
    "cashback ที่ได้รับหายไปจาก wallet",

    # General inquiry
    "ร้านเปิดกี่โมงถึงกี่โมงคะ",
    "ติดต่อ call center ได้ทางไหนบ้าง",
    "สินค้าตัวนี้มีของในสต็อกไหม",
    "อยากรู้ว่าสินค้าผลิตในประเทศอะไร",
    "มีหน้าร้านให้ไปดูสินค้าได้ไหมครับ",
]

# ── Build DataFrame ───────────────────────────────────────────
NUM_SESSIONS = 80
NUM_MESSAGES = 400

rows = []
for i in range(1, NUM_SESSIONS + 1):
    session_id = f"SESSION_{i:04d}"
    # Each session has 3–8 messages
    n = random.randint(3, 8)
    for _ in range(n):
        rows.append({
            "session_id": session_id,
            "message": random.choice(MESSAGES),
        })
    if len(rows) >= NUM_MESSAGES:
        break

df = pd.DataFrame(rows)[["message", "session_id"]]  # column order matches config

# ── Save ──────────────────────────────────────────────────────
out_path = Path(__file__).parent.parent / "artifacts" / "raw" / "chat_data.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"Saved {len(df):,} rows → {out_path}")
print(f"Columns : {list(df.columns)}")
print(f"Sessions: {df['session_id'].nunique():,}")
print("\nSample:")
print(df.head(8).to_string(index=False))
