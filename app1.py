import gzip
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
from threading import Lock

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a Tableau sales dashboard analyst.

Rules:
- Use ONLY facts present in the DASHBOARD_DATA section. Do not invent metrics, regions, products, or dates.
- If something is not in the data, say clearly that the current extract does not show it.
- Prefer exact figures from the data. Replace $ with PKR when reporting currency if the data uses those symbols.
- For follow-ups, stay consistent with your earlier answers in this conversation when they refer to the same metrics.
- Be concise unless the user asks for detail."""

MAX_DASHBOARD_CHARS = 120_000
MAX_HISTORY_MESSAGES = 6  # 6 Q&A turns (user + assistant pairs)

# Store dashboard data in memory
dashboard_data_store = {}
# Per (dashboard, session_id): list of {role, content} — user content is the question text only (no data blob)
chat_history_store = {}
_chat_lock = Lock()


def _session_key(dashboard: str, session_id: str) -> str:
    return f"{dashboard}\t{session_id}"


def compact_dashboard_text(data, max_chars: int = MAX_DASHBOARD_CHARS) -> str:
    """Turn stored worksheet payload into readable text; shrink if over budget."""
    if data is None:
        return ""
    if isinstance(data, str):
        return data if len(data) <= max_chars else data[: max_chars - 200] + "\n\n[... truncated: extract exceeded size limit ...]"
    if not isinstance(data, list):
        raw = str(data)
        return raw if len(raw) <= max_chars else raw[: max_chars - 200] + "\n\n[... truncated ...]"

    def build(max_rows_per_sheet: int) -> str:
        parts = []
        for ws in data:
            if not isinstance(ws, dict):
                parts.append(str(ws))
                continue
            name = ws.get("worksheet") or "Worksheet"
            cols = ws.get("columns") or []
            rows = ws.get("data") or []
            if max_rows_per_sheet > 0:
                rows = rows[:max_rows_per_sheet]
            header = " | ".join(str(c) for c in cols)
            lines = [f"=== {name} ===", header]
            for row in rows:
                lines.append(" | ".join(str(c) for c in row))
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    max_rows = 50_000
    text = build(max_rows)
    while len(text) > max_chars and max_rows > 10:
        max_rows = max(max_rows // 2, 10)
        text = build(max_rows)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n\n[... truncated: narrow filters or ask about a specific sheet ...]"

@app.route('/api/receiveData', methods=['POST'])
def receive_data():
    data = request.get_json()
    print("✅ Received Data:")
    return jsonify({"status": "success"}), 200


@app.route("/api/store_data", methods=["POST"])
def store_data():
    try:
        # 🔹 Read and decompress gzip data
        compressed = request.get_data()
        decompressed = gzip.decompress(compressed)
        content = json.loads(decompressed.decode("utf-8"))

        dashboard = content.get("dashboard")
        data = content.get("data")

        dashboard_data_store[dashboard] = data
        # print(dashboard_data_store[dashboard])
        print(f"✅ Received {len(str(data))} characters of data")
        # print(data)


        return jsonify({"status": "success"})
    except Exception as e:
        print("❌ Error in store_data:", e)
        return jsonify({"error": str(e)}), 400

def _build_user_message(
    dashboard_text: str,
    question: str,
    current_period,
    reference_period,
    is_narrative: bool,
) -> str:
    ctx_lines = []
    if current_period:
        ctx_lines.append(f"Current period: {current_period}")
    if reference_period:
        ctx_lines.append(f"Reference period: {reference_period}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "(not specified)"

    if is_narrative:
        task = (
            "Generate a concise executive summary using ONLY the data below. "
            "Cover revenue trends, top and weak areas, and notable changes. "
            "Use clear language for business stakeholders. Include time periods where applicable. "
            "For comparisons use + or − with percentages (e.g. +12.5%, −7.3%)."
        )
    else:
        task = (
            "Answer the question using ONLY the data below. "
            "For month-specific questions, use monthly rows; do not use YTD summary rows unless the user asks for YTD. "
            "Give the direct answer without explaining calculation steps."
        )

    return (
        f"--- DASHBOARD_DATA ---\n{dashboard_text}\n"
        f"--- CONTEXT ---\n{context_block}\n"
        f"--- TASK ---\n{task}\n"
        f"--- QUESTION ---\n{question.strip()}"
    )


@app.route("/api/reset_session", methods=["POST"])
def reset_session():
    try:
        body = request.json or {}
        dashboard = body.get("dashboard")
        session_id = body.get("session_id")
        if not dashboard or not session_id:
            return jsonify({"error": "dashboard and session_id required"}), 400
        key = _session_key(dashboard, session_id)
        with _chat_lock:
            chat_history_store.pop(key, None)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        content = request.json or {}
        question = (content.get('question') or '').strip()
        dashboard = content.get('dashboard')
        session_id = (content.get('session_id') or '').strip() or "default"

        current_period = content.get('current_period')
        reference_period = content.get('reference_period')

        if not dashboard or dashboard not in dashboard_data_store:
            return jsonify({"error": "Dashboard data not found"}), 400

        data = dashboard_data_store[dashboard]
        if not data:
            return jsonify({"error": "No data available for this dashboard."}), 400

        dashboard_text = compact_dashboard_text(data)
        data_for_regex = dashboard_text

        if not reference_period:
            reference_period_match = re.search(
                r"Reference (Month|Period):\s*(\w+\s+\d{4})", data_for_regex
            )
            reference_period = (
                reference_period_match.group(2) if reference_period_match else None
            )

        print(f"Dashboard text length (compact): {len(dashboard_text)}")

        q_lower = question.lower()
        is_narrative_request = (
            q_lower
            in (
                "generate summary",
                "give me a summary",
                "provide a summary narrative of the current dashboard data.",
                "dashboard summary",
                "summarize",
            )
            or "summary" in q_lower
        )

        user_message = _build_user_message(
            dashboard_text,
            question,
            current_period,
            reference_period,
            is_narrative_request,
        )

        key = _session_key(dashboard, session_id)
        with _chat_lock:
            prior = list(chat_history_store.get(key, []))

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in prior:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )

        answer = response.choices[0].message.content
        answer = answer.replace("$", "PKR ")

        with _chat_lock:
            hist = chat_history_store.setdefault(key, [])
            hist.append({"role": "user", "content": question})
            hist.append({"role": "assistant", "content": answer})
            if len(hist) > MAX_HISTORY_MESSAGES:
                chat_history_store[key] = hist[-MAX_HISTORY_MESSAGES:]

        return jsonify({"answer": answer, "session_id": session_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run()
