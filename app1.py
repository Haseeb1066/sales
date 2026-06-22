import gzip
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

import sales_core as core

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.route("/api/receiveData", methods=["POST"])
def receive_data():
    request.get_json()
    print("✅ Received Data:")
    return jsonify({"status": "success"}), 200


@app.route("/api/store_data", methods=["POST"])
def store_data():
    try:
        compressed = request.get_data()
        decompressed = gzip.decompress(compressed)
        content = json.loads(decompressed.decode("utf-8"))

        dashboard = content.get("dashboard")
        data = content.get("data")

        if not dashboard:
            return jsonify({"error": "dashboard name is required"}), 400
        if data is None:
            return jsonify({"error": "data is required"}), 400

        core.dashboard_data_store[dashboard] = data
        print(f"✅ Stored dashboard '{dashboard}' ({len(str(data))} chars)")

        return jsonify({"status": "success"})
    except Exception as e:
        print("❌ Error in store_data:", e)
        return jsonify({"error": str(e)}), 400


@app.route("/api/dashboards", methods=["GET"])
def list_dashboards():
    return jsonify({"dashboards": core.list_dashboards()})


@app.route("/api/dashboard/<path:name>/worksheets", methods=["GET"])
def dashboard_worksheets(name):
    if name not in core.dashboard_data_store:
        return jsonify({"error": "Dashboard data not found"}), 404
    return jsonify({"dashboard": name, "worksheets": core.list_worksheets(name)})


@app.route("/api/reset_session", methods=["POST"])
def reset_session():
    try:
        body = request.json or {}
        dashboard = body.get("dashboard")
        session_id = body.get("session_id")
        if not dashboard or not session_id:
            return jsonify({"error": "dashboard and session_id required"}), 400
        core.reset_session(dashboard, session_id)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        content = request.json or {}
        result = core.ask_dashboard(
            client,
            dashboard=content.get("dashboard"),
            question=(content.get("question") or "").strip(),
            session_id=(content.get("session_id") or "").strip() or "default",
            current_period=content.get("current_period"),
            reference_period=content.get("reference_period"),
        )
        if "error" in result:
            status = 400 if result["error"] in (
                "Dashboard data not found",
                "No data available for this dashboard.",
                "question is required",
            ) else 500
            return jsonify(result), status
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
