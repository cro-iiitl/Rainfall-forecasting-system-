"""
7-day rainfall prediction using the trained project model and live weather data.

This script combines the useful parts of the notebook:
- loads model.pkl and scaler.pkl
- preserves the model's exact feature order
- fetches live forecast parameters from a weather API
- converts wind speed/direction into u10 and v10 components
- fills ERA5-only fields from nearby historical district data
- predicts rainfall for the next 7 days

Examples:
    python3 rainfall_7day_forecast.py --district Agra
    python3 rainfall_7day_forecast.py --lat 27.1767 --lon 78.0081
    python3 rainfall_7day_forecast.py --district Lucknow --output lucknow_forecast.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "features.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "up_daily_weather_with_latlon.csv"

FALLBACK_FEATURES = [
    "lat",
    "lon",
    "sp",
    "tcc",
    "u10",
    "v10",
    "t2m",
    "d2m",
    "lcc",
    "viwve",
    "viwvn",
]

LIVE_HOURLY_FIELDS = [
    "temperature_2m",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]


def haversine_km(lat1: float, lon1: float, lat2: Iterable[float], lon2: Iterable[float]) -> np.ndarray:
    """Vectorized distance from one point to many points."""
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def wind_components(speed_ms: float, direction_deg: float) -> tuple[float, float]:
    """Convert meteorological wind direction into u/v components."""
    radians = math.radians(direction_deg)
    u10 = -speed_ms * math.sin(radians)
    v10 = -speed_ms * math.cos(radians)
    return u10, v10


def load_feature_names(scaler) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    try:
        return list(joblib.load(FEATURES_PATH))
    except Exception:
        return FALLBACK_FEATURES


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = load_feature_names(scaler)
    return model, scaler, features


def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed data file not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["lat", "lon"])
    df["month"] = df["date"].dt.month
    return df


def resolve_location(df: pd.DataFrame, district: str | None, lat: float | None, lon: float | None) -> tuple[float, float, str]:
    if district:
        district_rows = df[df["district"].str.casefold() == district.casefold()]
        if district_rows.empty:
            choices = ", ".join(sorted(df["district"].dropna().unique())[:20])
            raise ValueError(f"District '{district}' was not found. Example districts: {choices}")

        row = district_rows.iloc[0]
        return float(row["lat"]), float(row["lon"]), str(row["district"])

    if lat is None or lon is None:
        raise ValueError("Provide either --district or both --lat and --lon.")

    return float(lat), float(lon), "custom_location"


def nearest_history(df: pd.DataFrame, lat: float, lon: float, month: int) -> pd.Series:
    work = df.copy()
    work["distance_km"] = haversine_km(lat, lon, work["lat"], work["lon"])
    nearest_district = work.sort_values("distance_km").iloc[0]["district"]
    nearest_month = work[(work["district"] == nearest_district) & (work["month"] == month)]

    if nearest_month.empty:
        nearest_month = work[work["district"] == nearest_district]

    return nearest_month[["viwve", "viwvn"]].mean(numeric_only=True)


def fetch_open_meteo_7day(lat: float, lon: float) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(LIVE_HOURLY_FIELDS),
        "forecast_days": 7,
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not fetch live weather data from Open-Meteo. "
            "Check your internet connection and try again."
        ) from exc
    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError(f"Weather API did not return hourly data: {payload}")

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.date

    daily = (
        hourly.groupby("date", as_index=False)
        .agg(
            temperature_2m=("temperature_2m", "mean"),
            dew_point_2m=("dew_point_2m", "mean"),
            surface_pressure=("surface_pressure", "mean"),
            cloud_cover=("cloud_cover", "mean"),
            cloud_cover_low=("cloud_cover_low", "mean"),
            wind_speed_10m=("wind_speed_10m", "mean"),
            wind_direction_10m=("wind_direction_10m", "mean"),
            api_precipitation_mm=("precipitation", "sum"),
        )
        .head(7)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def build_model_input(live_daily: pd.DataFrame, history: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    rows = []

    for _, day in live_daily.iterrows():
        u10, v10 = wind_components(float(day["wind_speed_10m"]), float(day["wind_direction_10m"]))
        historical = nearest_history(history, lat, lon, int(day["date"].month))

        rows.append(
            {
                "date": day["date"].date().isoformat(),
                "lat": lat,
                "lon": lon,
                "sp": float(day["surface_pressure"]),
                "tcc": float(day["cloud_cover"]),
                "u10": u10,
                "v10": v10,
                "t2m": float(day["temperature_2m"]),
                # The saved model was trained with d2m still in Kelvin.
                "d2m": float(day["dew_point_2m"]) + 273.15,
                "lcc": float(day["cloud_cover_low"]) / 100.0,
                "viwve": float(historical["viwve"]),
                "viwvn": float(historical["viwvn"]),
                "api_precipitation_mm": float(day["api_precipitation_mm"]),
            }
        )

    return pd.DataFrame(rows)


def predict_7day_rainfall(district: str | None = None, lat: float | None = None, lon: float | None = None) -> pd.DataFrame:
    model, scaler, features = load_artifacts()
    history = load_training_data()
    resolved_lat, resolved_lon, location_name = resolve_location(history, district, lat, lon)

    live_daily = fetch_open_meteo_7day(resolved_lat, resolved_lon)
    input_df = build_model_input(live_daily, history, resolved_lat, resolved_lon)

    X = input_df.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    output = input_df[
        [
            "date",
            "t2m",
            "sp",
            "tcc",
            "u10",
            "v10",
            "lcc",
            "api_precipitation_mm",
        ]
    ].copy()
    output.insert(0, "location", location_name)
    output.insert(1, "lat", round(resolved_lat, 6))
    output.insert(2, "lon", round(resolved_lon, 6))
    output["predicted_rainfall_mm"] = np.maximum(predictions, 0).round(3)
    output["api_precipitation_mm"] = output["api_precipitation_mm"].round(3)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict rainfall for the next 7 days using live weather API values.")
    parser.add_argument("--district", help="Uttar Pradesh district name, for example: Agra")
    parser.add_argument("--lat", type=float, help="Latitude for custom location prediction")
    parser.add_argument("--lon", type=float, help="Longitude for custom location prediction")
    parser.add_argument("--output", help="Optional CSV path to save the 7-day forecast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecast = predict_7day_rainfall(district=args.district, lat=args.lat, lon=args.lon)

    print("\n7-day rainfall forecast")
    print(forecast.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        forecast.to_csv(output_path, index=False)
        print(f"\nSaved forecast to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
