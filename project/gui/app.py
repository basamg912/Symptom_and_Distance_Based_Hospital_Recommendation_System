"""
Flask web application for the Hospital Recommendation System.

Run with:
    python -m project.gui.app
"""

import os
import sys

# Ensure the repository root is on sys.path so 'project.*' imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, render_template, request, jsonify
from project.gui.service import recommend_hospitals

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


@app.route("/")
def index():
    """Serve the main single-page application."""
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    POST /api/recommend
    Body: {"symptom": "증상 텍스트..."}
    Returns JSON with extracted_dept and hospitals list.
    """
    data = request.get_json(force=True)
    symptom = data.get("symptom", "").strip()

    if not symptom:
        return jsonify({"error": "증상을 입력해주세요."}), 400

    try:
        result = recommend_hospitals(symptom, top_k=10)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
