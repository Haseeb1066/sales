from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Store dashboard data
dashboard_data_store = {}

@app.route('/api/receiveData', methods=['POST'])
def receive_data():
    data = request.get_json()
    print("✅ Received Data:")
    return jsonify({"status": "success"}), 200

@app.route("/api/store_data", methods=["POST"])
def store_data():
    content = request.get_json()
    dashboard = content.get("dashboard")
    data = content.get("data")
    dashboard_data_store[dashboard] = data
    return jsonify({"status": "success"})

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        content = request.json
        question = content.get('question', '')
        dashboard = content.get('dashboard', None)

        if not dashboard or dashboard not in dashboard_data_store:
            return jsonify({"error": "Dashboard data not found"}), 400

        data = dashboard_data_store[dashboard]
        if not data:
            return jsonify({"error": "No data available for this dashboard."}), 400

        # Truncate data if too large
        import json
        raw_data_text = json.dumps(data)
        if len(raw_data_text) > 6000:
            raw_data_text = raw_data_text[:6000] + "\n\n...(truncated)"

        # Detect narrative request
        is_narrative_request = (
            question.strip().lower() in [
                "generate summary",
                "give me a summary",
                "provide a summary narrative of the current dashboard data.",
                "dashboard summary",
                "summarize"
            ] or "summary" in question.lower()
        )

        if is_narrative_request:
            prompt = (
                f"You are a Tableau dashboard analyst. Given the following data:\n\n"
                f"{raw_data_text}\n\n"
                "Generate a concise executive summary or narrative explaining key insights, trends, and patterns in simple language. "
                "Highlight sales, regional performance, and any outliers or notable changes."
            )
        else:
            prompt = (
                f"The user has asked a question based on this Tableau dashboard data:\n\n"
                f"{raw_data_text}\n\n"
                f"Question: {question}\n\n"
                f"Only provide the relevant answer without explaining how it was calculated."
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Tableau dashboard analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
