import json
import time
import logging
import ollama
from datetime import datetime
from pathlib import Path
from config import OLLAMA_MODEL, TEXT_DELIMITER, MAX_DOC_CHARS

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)


def build_prompt(docs_str: str) -> str:
    """
    Topic labeling prompt tuned for typhoon2.1-gemma3-4b.

    Design decisions:
    - Written in Thai for better Thai output quality
      (Typhoon responds more accurately to Thai instructions for Thai tasks)
    - Explicit JSON-only output instruction (Gemma3 tends to add extra text)
    - Short, unambiguous rules — local models need stricter guidance than API models
    - "passage: " prefix not needed here — this is generative, not embedding
    """
    return f"""คุณกำลังวิเคราะห์ข้อความของลูกค้าจาก Chat ของธุรกิจไทย
ข้อความเหล่านี้เป็นภาษาไทย ภาษาอังกฤษ หรือผสมกัน

ข้อความตัวแทนของกลุ่มนี้ คั่นด้วย {TEXT_DELIMITER}:
{docs_str}

วิเคราะห์ว่าลูกค้ากลุ่มนี้ต้องการอะไร หรือมีปัญหาเรื่องอะไร

กฎ:
- ชื่อหัวข้อ: 3-6 คำภาษาไทย สั้น กระชับ สื่อ Intent ลูกค้า
  ตัวอย่าง: "สอบถามสถานะการจัดส่ง", "ขอเปลี่ยน/คืนสินค้า", "ปัญหาการชำระเงิน"
- คำอธิบาย: 1 ประโยคภาษาไทย อธิบาย Pattern ลูกค้ากลุ่มนี้
- ถ้าไม่สามารถระบุได้: ใช้ "ไม่สามารถระบุหัวข้อได้"

ตอบด้วย JSON เท่านั้น ห้ามมีข้อความอื่น:
{{"topic_name": "<ชื่อหัวข้อ>", "topic_description": "<คำอธิบาย>"}}"""


def call_ollama(prompt: str, retries: int = 3) -> str:
    """
    Call local Typhoon model via Ollama.

    temperature=0 ensures deterministic output — same representative
    documents always produce the same topic label across runs.
    num_predict limits output length — topic labels are always short.
    """
    for attempt in range(retries):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.0,
                    "num_predict": 200,
                }
            )
            return response["message"]["content"].strip()

        except Exception as e:
            logging.error(f"Ollama error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)

    return ""


def parse_json(raw: str) -> dict:
    """
    Parse JSON from model output.
    Handles markdown fences and extra surrounding text — common in local models.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()

    # Attempt direct parse
    try:
        parsed = json.loads(clean)
        # Model sometimes wraps the object in an array — unwrap if so
        if isinstance(parsed, list):
            parsed = parsed[0] if parsed and isinstance(parsed[0], dict) else {}
        return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON by brace boundaries
    start = clean.find("{")
    end   = clean.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(clean[start:end])
        except json.JSONDecodeError:
            pass

    logging.warning(f"JSON parse failed. Raw output: {raw}")
    return {"error": "parse_failed", "raw": raw}


def label_topics(rep_docs: dict) -> dict:
    """
    Label each topic cluster by sending its representative documents
    to the local Typhoon model via Ollama.

    Only 3-5 representative documents are sent per topic —
    not the full corpus. This keeps inference fast and focused.

    rep_docs format: {topic_id (int): [doc1, doc2, doc3]}
    Returns        : {topic_id (str): {"topic_name": ..., "topic_description": ...}}
    """
    Path("../data/topics").mkdir(parents=True, exist_ok=True)
    formatted_dt = datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    results      = {}
    total        = len(rep_docs)

    for i, (topic_id, docs_list) in enumerate(rep_docs.items()):
        logging.info(f"Labeling topic {topic_id} ({i+1}/{total})...")

        # Truncate each doc to MAX_DOC_CHARS to control context length
        truncated = [str(d)[:MAX_DOC_CHARS] for d in docs_list]
        docs_str  = TEXT_DELIMITER.join(truncated)
        prompt    = build_prompt(docs_str)
        raw       = call_ollama(prompt)

        if not raw:
            results[str(topic_id)] = {"error": "empty_response"}
            continue

        if "ไม่สามารถระบุหัวข้อได้" in raw:
            results[str(topic_id)] = {
                "topic_name": "ไม่สามารถระบุหัวข้อได้",
                "topic_description": ""
            }
        else:
            results[str(topic_id)] = parse_json(raw)

    # Save with Thai characters preserved
    # ensure_ascii=False is mandatory — otherwise Thai becomes \uXXXX escape sequences
    out_path = f"../data/topics/labels_{formatted_dt}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"Labeled {len(results)} topics → {out_path}")
    return results