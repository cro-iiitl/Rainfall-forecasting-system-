"""
Flask API backend for the UP Rainfall Prediction frontend.
Wraps latest_rainfall_prediction.py and exposes REST endpoints.

Run:
    python3 app.py

Config (all optional, via environment variables):
    HOST            default 0.0.0.0
    PORT            default 5050
    FLASK_DEBUG     default 0  (set to 1 only for local development)
    FORECAST_CACHE_TTL_SECONDS   default 900 (15 min) — how long a
        district/lat-lon forecast response is reused before re-hitting
        Open-Meteo + re-running the models.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# ── import project logic ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import latest_rainfall_prediction as lrp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rainfall_api")

app = Flask(__name__)
CORS(app)

# ── config ───────────────────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5050"))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
FORECAST_CACHE_TTL_SECONDS = int(os.environ.get("FORECAST_CACHE_TTL_SECONDS", "900"))

# Loose bounding box around Uttar Pradesh (+ margin) to reject obviously
# wrong lat/lon before they reach the model / an external API call.
UP_LAT_RANGE = (22.0, 31.5)
UP_LON_RANGE = (76.5, 85.0)

# ── cached district list (loaded once, not per-request) ────────────────────
_DISTRICTS: list[str] | None = None
_DISTRICTS_LOCK = Lock()

# ── small TTL cache for /api/predict, keyed by the resolved request params ─
_forecast_cache: dict[tuple, tuple[float, dict]] = {}
_forecast_cache_lock = Lock()


def get_districts() -> list[str]:
    global _DISTRICTS
    if _DISTRICTS is None:
        with _DISTRICTS_LOCK:
            if _DISTRICTS is None:  # re-check inside the lock
                try:
                    history = lrp.load_raw_history()
                    _DISTRICTS = sorted(history["district"].dropna().unique().tolist())
                except Exception:
                    logger.exception("Failed to load district list")
                    _DISTRICTS = []
    return _DISTRICTS


def _parse_float(value, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number.")


def _validate_district(district) -> str | None:
    if district is None:
        return None
    if not isinstance(district, str):
        raise ValueError("'district' must be a string.")
    district = district.strip()
    if not district:
        return None
    if len(district) > 100:
        raise ValueError("'district' is too long.")
    return district


def _validate_latlon(lat: float | None, lon: float | None) -> None:
    if lat is None and lon is None:
        return
    if lat is None or lon is None:
        raise ValueError("Provide both 'lat' and 'lon', not just one.")
    if not (UP_LAT_RANGE[0] <= lat <= UP_LAT_RANGE[1]) or not (UP_LON_RANGE[0] <= lon <= UP_LON_RANGE[1]):
        raise ValueError(
            f"'lat'/'lon' look outside Uttar Pradesh "
            f"(expected roughly lat {UP_LAT_RANGE}, lon {UP_LON_RANGE})."
        )


def _cached_predict(district: str | None, lat: float | None, lon: float | None) -> dict:
    cache_key = (district, lat, lon)
    now = time.monotonic()

    with _forecast_cache_lock:
        cached = _forecast_cache.get(cache_key)
        if cached and (now - cached[0]) < FORECAST_CACHE_TTL_SECONDS:
            return cached[1]

    forecast_df = lrp.predict_7day(district, lat, lon)
    records = forecast_df.to_dict(orient="records")
    
    for row in records:
        api_p = row.get("api_precipitation_mm", 0.0)
        pred = row.get("predicted_rainfall_mm", 0.0)
        cc = row.get("cloud_cover_pct", 0.0)
        
        diff = pred - api_p
        if diff > 10:
            row["model_reasoning"] = f"API underestimated. 15-year UP data shows {cc:.0f}% cloud cover heavily increases rain potential."
        elif diff > 3:
            row["model_reasoning"] = "ML model adjusted forecast upwards based on local historical moisture trends."
        elif diff < -5:
            row["model_reasoning"] = "API overestimated. Local history dampens this prediction."
        else:
            row["model_reasoning"] = "Consistent with global API."

    payload = {"forecast": records}

    with _forecast_cache_lock:
        _forecast_cache[cache_key] = (now, payload)

    return payload


# ── routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/districts")
def api_districts():
    return jsonify({"districts": get_districts()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        district = _validate_district(data.get("district"))
        lat = _parse_float(data.get("lat"), "lat")
        lon = _parse_float(data.get("lon"), "lon")
        _validate_latlon(lat, lon)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not district and (lat is None or lon is None):
        return jsonify({"error": "Provide 'district' or both 'lat' and 'lon'."}), 400

    try:
        payload = _cached_predict(district, lat, lon)
        return jsonify(payload)
    except FileNotFoundError as exc:
        logger.warning("Model artifacts missing: %s", exc)
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        # Live weather API unavailable/unreachable after retries.
        logger.warning("Upstream weather API failure: %s", exc)
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        logger.exception("Unexpected prediction failure")
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
    logger.info("UP Rainfall Prediction API starting on http://%s:%s (debug=%s)", HOST, PORT, DEBUG)
    app.run(host=HOST, port=PORT, debug=DEBUG)
