import gzip
import json
from flask import request
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Store dashboard data in memory
dashboard_data_store = {}

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

@app.route('/api/ask', methods=['POST'])
def ask():
    try:
        content = request.json
        question = content.get('question', '')
        dashboard = content.get('dashboard', None)

        # ✅ Accept both current and reference periods
        current_period = content.get('current_period', None)
        reference_period = content.get('reference_period', None)

        if not dashboard or dashboard not in dashboard_data_store:
            return jsonify({"error": "Dashboard data not found"}), 400

        data = dashboard_data_store[dashboard]
        if not data:
            return jsonify({"error": "No data available for this dashboard."}), 400

        data_text = "\n".join([str(item) for item in data])

        # ✅ Fallback for reference_period (if not passed)
        if not reference_period:
            reference_period_match = re.search(r'Reference (Month|Period):\s*(\w+\s+\d{4})', data_text)
            reference_period = reference_period_match.group(2) if reference_period_match else None

        print(f"Total data length: {len(str(data_text))}")

        is_narrative_request = (
            question.strip().lower() in [
                "generate summary",
                "give me a summary",
                "provide a summary narrative of the current dashboard data.",
                "dashboard summary",
                "summarize"
            ] or "summary" in question.lower()
        )

        # ✅ Build prompt with both periods
        if is_narrative_request:
            prompt = (
                f"You are a Tableau sales dashboard analyst. Given the following extracted dashboard data:\n\n"
                f"{data}\n\n"
                f"Generate a concise and professional executive summary. Focus on key sales insights, including revenue trends, "
                f"top-performing regions or products, underperforming areas, and any significant changes or outliers. "
                f"Use clear and simple language suitable for business stakeholders. "
                f"Include the time period in each reported metric where applicable. "
                f"{'The current period is ' + current_period + '. ' if current_period else ''}"
                f"{'The reference period is ' + reference_period + '.' if reference_period else ''}"
            )
        else:
            prompt = (
                f"The user has asked a question based on this Tableau dashboard data:\n\n"
                f"{data_text}\n\n"
                f"{'The current period is ' + current_period + '. ' if current_period else ''}"
                f"{'The comparison is made against the reference period ' + reference_period + '.' if reference_period else ''}\n"
                f"Question: {question}\n\n"
                f"Only provide the relevant answer without explaining how it was calculated. "
                f"Always include both time periods in the sales figures or comparisons you provide."
                f"Always include the current and reference period (month and year) in every metric label, such as: 'Current Sales (May 2025)' and 'Reference Sales (April 2025)'."
            )

        # 🔹 Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Tableau dashboard analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # 🔹 Extract GPT response
        answer = response.choices[0].message.content

        # ✅ Replace $ with PKR (no conversion)
        answer = answer.replace("$", "PKR ")

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run()
