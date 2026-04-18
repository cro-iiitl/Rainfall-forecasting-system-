"""
Flask API backend for the UP Rainfall Prediction frontend.
Wraps latest_rainfall_prediction.py and exposes REST endpoints.

Run:
    python3 app.py
"""

from __future__ import annotations

import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ── import project logic ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import latest_rainfall_prediction as lrp

app = Flask(__name__)
CORS(app)

# ── cached district list ────────────────────────────────────────────────────
_DISTRICTS: list[str] | None = None


def get_districts() -> list[str]:
    global _DISTRICTS
    if _DISTRICTS is None:
        try:
            history = lrp.load_raw_history()
            _DISTRICTS = sorted(history["district"].dropna().unique().tolist())
        except Exception:
            _DISTRICTS = []
    return _DISTRICTS


# ── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/districts")
def api_districts():
    return jsonify({"districts": get_districts()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}

    district: str | None = data.get("district") or None
    lat: float | None = data.get("lat")
    lon: float | None = data.get("lon")

    if lat is not None:
        try:
            lat = float(lat)
        except (TypeError, ValueError):
            lat = None
    if lon is not None:
        try:
            lon = float(lon)
        except (TypeError, ValueError):
            lon = None

    if not district and (lat is None or lon is None):
        return jsonify({"error": "Provide 'district' or both 'lat' and 'lon'."}), 400

    try:
        forecast_df = lrp.predict_7day(district, lat, lon)
        records = forecast_df.to_dict(orient="records")
        return jsonify({"forecast": records})
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/meta")
def api_meta():
    """Return training metadata if available."""
    try:
        import joblib
        meta = joblib.load(lrp.META_PATH)
        return jsonify(meta)
    except Exception:
        return jsonify({}), 204


if __name__ == "__main__":
    print("🌧  UP Rainfall Prediction API running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=True)
