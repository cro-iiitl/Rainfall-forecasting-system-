"""
Merged rainfall prediction pipeline for the UP rainfall project.

This file combines:
- your current project assets: processed CSV with lat/lon, model.pkl, scaler.pkl
- the attached notebook's stronger feature engineering and seasonal models
- live 7-day weather values from Open-Meteo

Main usage:
    python3 merged_rainfall_prediction.py predict --district Agra
    python3 merged_rainfall_prediction.py predict --lat 27.1767 --lon 78.0081
    python3 merged_rainfall_prediction.py train

The first prediction will use enhanced seasonal models if they have been trained.
If not, it falls back to the existing legacy model.pkl/scaler.pkl so prediction
still works immediately.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError as exc:  # pragma: no cover - handled at runtime
    XGBClassifier = None
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc
else:
    XGBOOST_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "up_daily_weather_with_latlon.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

LEGACY_MODEL_PATH = BASE_DIR / "model.pkl"
LEGACY_SCALER_PATH = BASE_DIR / "scaler.pkl"

META_PATH = ARTIFACT_DIR / "rainfall_pipeline_meta.pkl"
BEST_REG_PATH = ARTIFACT_DIR / "best_regression_model.pkl"
BEST_CLS_PATH = ARTIFACT_DIR / "best_classification_model.pkl"
DRY_MODEL_PATH = ARTIFACT_DIR / "xgb_reg_dry.pkl"
MONSOON_MODEL_PATH = ARTIFACT_DIR / "xgb_reg_monsoon.pkl"
DISTRICT_ENCODER_PATH = ARTIFACT_DIR / "district_encoder.pkl"

ENHANCED_FEATURES = [
    "sp",
    "tcc",
    "u10",
    "v10",
    "t2m",
    "d2m",
    "lcc",
    "viwve",
    "viwvn",
    "dewpoint_depression",
    "wind_speed",
    "moisture_flux",
    "rain_lag_1",
    "rain_lag_2",
    "rain_lag_3",
    "rain_lag_7",
    "rain_roll3",
    "rain_roll7",
    "month_sin",
    "month_cos",
    "is_monsoon",
    "district_enc",
]

LEGACY_FEATURES = [
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


@dataclass
class Location:
    name: str
    lat: float
    lon: float
    model_district: str


def haversine_km(lat1: float, lon1: float, lat2: Iterable[float], lon2: Iterable[float]) -> np.ndarray:
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
    radians = math.radians(direction_deg)
    return -speed_ms * math.sin(radians), -speed_ms * math.cos(radians)


def load_raw_history() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.dropna(subset=["district", "date", "lat", "lon", "rainfall_mm"])
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    return df


def normalize_weather_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df["t2m"].mean() > 100:
        df["t2m"] = df["t2m"] - 273.15
    if df["d2m"].mean() > 100:
        df["d2m"] = df["d2m"] - 273.15
    if df["sp"].mean() > 2000:
        df["sp"] = df["sp"] / 100.0
    if df["tcc"].max() <= 1.5:
        df["tcc"] = df["tcc"] * 100.0

    return df


def build_enhanced_training_frame() -> tuple[pd.DataFrame, LabelEncoder]:
    df = normalize_weather_units(load_raw_history())

    rainfall_cap = df["rainfall_mm"].quantile(0.99)
    df["rainfall_mm"] = df["rainfall_mm"].clip(upper=rainfall_cap)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dewpoint_depression"] = df["t2m"] - df["d2m"]
    df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    df["moisture_flux"] = np.sqrt(df["viwve"] ** 2 + df["viwvn"] ** 2)
    df["is_monsoon"] = df["month"].between(6, 9).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    for lag in [1, 2, 3, 7]:
        df[f"rain_lag_{lag}"] = df.groupby("district")["rainfall_mm"].shift(lag)

    df["rain_roll3"] = df.groupby("district")["rainfall_mm"].transform(lambda x: x.shift(1).rolling(3).mean())
    df["rain_roll7"] = df.groupby("district")["rainfall_mm"].transform(lambda x: x.shift(1).rolling(7).mean())
    df = df.dropna(subset=["rain_lag_1", "rain_lag_7", "rain_roll7"]).reset_index(drop=True)

    encoder = LabelEncoder()
    df["district_enc"] = encoder.fit_transform(df["district"])
    df["rain_occurred"] = (df["rainfall_mm"] > 0.1).astype(int)

    return df, encoder


def train_enhanced_models() -> None:
    if XGBRegressor is None or XGBClassifier is None:
        raise RuntimeError("xgboost is required for enhanced training.") from XGBOOST_IMPORT_ERROR

    ARTIFACT_DIR.mkdir(exist_ok=True)
    df, encoder = build_enhanced_training_frame()

    train = df[df["year"] <= 2022]
    test = df[df["year"] >= 2023]
    if train.empty or test.empty:
        cutoff = int(df["year"].quantile(0.8))
        train = df[df["year"] <= cutoff]
        test = df[df["year"] > cutoff]

    X_train = train[ENHANCED_FEATURES]
    X_test = test[ENHANCED_FEATURES]
    y_train_reg = train["rainfall_mm"]
    y_test_reg = test["rainfall_mm"]
    y_train_cls = train["rain_occurred"]
    y_test_cls = test["rain_occurred"]

    rf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    rf_reg.fit(X_train, y_train_reg)
    pred_rf_reg = np.clip(rf_reg.predict(X_test), 0, None)

    xgb_reg = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    xgb_reg.fit(X_train, y_train_reg)
    pred_xgb_reg = np.clip(xgb_reg.predict(X_test), 0, None)

    rf_cls = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
    )
    rf_cls.fit(X_train, y_train_cls)
    pred_rf_cls = rf_cls.predict(X_test)

    xgb_cls = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        eval_metric="logloss",
    )
    xgb_cls.fit(X_train, y_train_cls)
    pred_xgb_cls = xgb_cls.predict(X_test)

    rf_r2 = r2_score(y_test_reg, pred_rf_reg)
    xgb_r2 = r2_score(y_test_reg, pred_xgb_reg)
    rf_acc = accuracy_score(y_test_cls, pred_rf_cls)
    xgb_acc = accuracy_score(y_test_cls, pred_xgb_cls)

    best_reg = xgb_reg if xgb_r2 >= rf_r2 else rf_reg
    best_cls = xgb_cls if xgb_acc >= rf_acc else rf_cls

    train_dry = train[train["is_monsoon"] == 0]
    train_monsoon = train[train["is_monsoon"] == 1]

    xgb_dry = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    xgb_dry.fit(train_dry[ENHANCED_FEATURES], np.log1p(train_dry["rainfall_mm"]))

    xgb_monsoon = XGBRegressor(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=15,
        gamma=2,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    xgb_monsoon.fit(train_monsoon[ENHANCED_FEATURES], np.log1p(train_monsoon["rainfall_mm"]))

    meta = {
        "features": ENHANCED_FEATURES,
        "rainfall_cap": float(df["rainfall_mm"].max()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "rf_reg_mae": float(mean_absolute_error(y_test_reg, pred_rf_reg)),
        "rf_reg_r2": float(rf_r2),
        "xgb_reg_mae": float(mean_absolute_error(y_test_reg, pred_xgb_reg)),
        "xgb_reg_r2": float(xgb_r2),
        "rf_cls_accuracy": float(rf_acc),
        "xgb_cls_accuracy": float(xgb_acc),
    }

    joblib.dump(meta, META_PATH)
    joblib.dump(encoder, DISTRICT_ENCODER_PATH)
    joblib.dump(best_reg, BEST_REG_PATH)
    joblib.dump(best_cls, BEST_CLS_PATH)
    joblib.dump(xgb_dry, DRY_MODEL_PATH)
    joblib.dump(xgb_monsoon, MONSOON_MODEL_PATH)

    print("Enhanced models trained and saved in:", ARTIFACT_DIR)
    print(f"RF regression  MAE={meta['rf_reg_mae']:.2f}, R2={meta['rf_reg_r2']:.4f}")
    print(f"XGB regression MAE={meta['xgb_reg_mae']:.2f}, R2={meta['xgb_reg_r2']:.4f}")
    print(f"RF classifier  accuracy={meta['rf_cls_accuracy'] * 100:.2f}%")
    print(f"XGB classifier accuracy={meta['xgb_cls_accuracy'] * 100:.2f}%")


def resolve_location(history: pd.DataFrame, district: str | None, lat: float | None, lon: float | None) -> Location:
    districts = history[["district", "lat", "lon"]].drop_duplicates()

    if district:
        matched = districts[districts["district"].str.casefold() == district.casefold()]
        if matched.empty:
            examples = ", ".join(sorted(districts["district"].unique())[:20])
            raise ValueError(f"District '{district}' was not found. Example districts: {examples}")

        row = matched.iloc[0]
        return Location(str(row["district"]), float(row["lat"]), float(row["lon"]), str(row["district"]))

    if lat is None or lon is None:
        raise ValueError("Provide either --district or both --lat and --lon.")

    work = districts.copy()
    work["distance_km"] = haversine_km(float(lat), float(lon), work["lat"], work["lon"])
    nearest = work.sort_values("distance_km").iloc[0]
    return Location("custom_location", float(lat), float(lon), str(nearest["district"]))


def nearest_monthly_moisture(history: pd.DataFrame, location: Location, month: int) -> pd.Series:
    work = history.copy()
    work["distance_km"] = haversine_km(location.lat, location.lon, work["lat"], work["lon"])
    nearest_district = work.sort_values("distance_km").iloc[0]["district"]

    district_month = work[(work["district"] == nearest_district) & (work["date"].dt.month == month)]
    if district_month.empty:
        district_month = work[work["district"] == nearest_district]

    return district_month[["viwve", "viwvn"]].mean(numeric_only=True)


def latest_rain_memory(history: pd.DataFrame, district: str) -> list[float]:
    series = history[history["district"] == district].sort_values("date")["rainfall_mm"].tail(14).tolist()
    if len(series) < 7:
        series = ([0.0] * (7 - len(series))) + series
    return [float(x) for x in series]


def fetch_live_7day_weather(lat: float, lon: float) -> pd.DataFrame:
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
        raise RuntimeError("Could not fetch live weather data. Check your internet connection.") from exc

    payload = response.json()
    if "hourly" not in payload:
        raise RuntimeError(f"Weather API did not return hourly data: {payload}")

    hourly = pd.DataFrame(payload["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["date"] = hourly["time"].dt.date

    daily = (
        hourly.groupby("date", as_index=False)
        .agg(
            t2m=("temperature_2m", "mean"),
            t2m_min=("temperature_2m", "min"),
            t2m_max=("temperature_2m", "max"),
            d2m=("dew_point_2m", "mean"),
            sp=("surface_pressure", "mean"),
            tcc=("cloud_cover", "mean"),
            lcc_percent=("cloud_cover_low", "mean"),
            wind_speed_api=("wind_speed_10m", "mean"),
            wind_direction_api=("wind_direction_10m", "mean"),
            api_precipitation_mm=("precipitation", "sum"),
        )
        .head(7)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def district_code(encoder: LabelEncoder, district: str) -> int:
    if district in set(encoder.classes_):
        return int(encoder.transform([district])[0])
    return 0


def build_live_enhanced_features(
    live_daily: pd.DataFrame,
    history: pd.DataFrame,
    location: Location,
    encoder: LabelEncoder,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rain_memory = latest_rain_memory(history, location.model_district)
    feature_rows = []
    display_rows = []

    for _, day in live_daily.iterrows():
        u10, v10 = wind_components(float(day["wind_speed_api"]), float(day["wind_direction_api"]))
        moisture = nearest_monthly_moisture(history, location, int(day["date"].month))

        rain_lag_1 = rain_memory[-1]
        rain_lag_2 = rain_memory[-2]
        rain_lag_3 = rain_memory[-3]
        rain_lag_7 = rain_memory[-7]
        rain_roll3 = float(np.mean(rain_memory[-3:]))
        rain_roll7 = float(np.mean(rain_memory[-7:]))
        month = int(day["date"].month)

        row = {
            "sp": float(day["sp"]),
            "tcc": float(day["tcc"]),
            "u10": u10,
            "v10": v10,
            "t2m": float(day["t2m"]),
            "d2m": float(day["d2m"]),
            "lcc": float(day["lcc_percent"]) / 100.0,
            "viwve": float(moisture["viwve"]),
            "viwvn": float(moisture["viwvn"]),
            "dewpoint_depression": float(day["t2m"] - day["d2m"]),
            "wind_speed": float(math.sqrt(u10**2 + v10**2)),
            "moisture_flux": float(math.sqrt(float(moisture["viwve"]) ** 2 + float(moisture["viwvn"]) ** 2)),
            "rain_lag_1": rain_lag_1,
            "rain_lag_2": rain_lag_2,
            "rain_lag_3": rain_lag_3,
            "rain_lag_7": rain_lag_7,
            "rain_roll3": rain_roll3,
            "rain_roll7": rain_roll7,
            "month_sin": float(math.sin(2 * math.pi * month / 12)),
            "month_cos": float(math.cos(2 * math.pi * month / 12)),
            "is_monsoon": int(6 <= month <= 9),
            "district_enc": district_code(encoder, location.model_district),
        }

        feature_rows.append(row)
        display_rows.append(
            {
                "location": location.name,
                "model_district": location.model_district,
                "lat": round(location.lat, 6),
                "lon": round(location.lon, 6),
                "date": day["date"].date().isoformat(),
                "t2m_mean_c": round(float(day["t2m"]), 3),
                "t2m_min_c": round(float(day["t2m_min"]), 3),
                "t2m_max_c": round(float(day["t2m_max"]), 3),
                "d2m_mean_c": round(float(day["d2m"]), 3),
                "sp_hpa": round(float(day["sp"]), 3),
                "cloud_cover_pct": round(float(day["tcc"]), 3),
                "api_precipitation_mm": round(float(day["api_precipitation_mm"]), 3),
            }
        )

        # Temporary placeholder. The prediction loop replaces this after each day.
        rain_memory.append(rain_lag_1)

    return pd.DataFrame(feature_rows), pd.DataFrame(display_rows)


def predict_with_enhanced_models(live_daily: pd.DataFrame, history: pd.DataFrame, location: Location) -> pd.DataFrame:
    encoder = joblib.load(DISTRICT_ENCODER_PATH)
    dry_model = joblib.load(DRY_MODEL_PATH)
    monsoon_model = joblib.load(MONSOON_MODEL_PATH)
    classifier = joblib.load(BEST_CLS_PATH) if BEST_CLS_PATH.exists() else None

    rain_memory = latest_rain_memory(history, location.model_district)
    rows = []

    for _, day in live_daily.iterrows():
        one_day = pd.DataFrame([day])
        features_df, display_df = build_live_enhanced_features(one_day, history, location, encoder)

        features_df.loc[0, "rain_lag_1"] = rain_memory[-1]
        features_df.loc[0, "rain_lag_2"] = rain_memory[-2]
        features_df.loc[0, "rain_lag_3"] = rain_memory[-3]
        features_df.loc[0, "rain_lag_7"] = rain_memory[-7]
        features_df.loc[0, "rain_roll3"] = float(np.mean(rain_memory[-3:]))
        features_df.loc[0, "rain_roll7"] = float(np.mean(rain_memory[-7:]))

        X = features_df.reindex(columns=ENHANCED_FEATURES, fill_value=0)
        model = monsoon_model if int(X.loc[0, "is_monsoon"]) == 1 else dry_model
        predicted = float(np.expm1(model.predict(X)[0]))
        predicted = max(predicted, 0.0)

        display = display_df.iloc[0].to_dict()
        display["predicted_rainfall_mm"] = round(predicted, 3)
        if classifier is not None and hasattr(classifier, "predict_proba"):
            display["rain_probability_pct"] = round(float(classifier.predict_proba(X)[0, 1]) * 100, 2)

        rows.append(display)
        rain_memory.append(predicted)

    return pd.DataFrame(rows)


def legacy_feature_names(scaler) -> list[str]:
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    return LEGACY_FEATURES


def predict_with_legacy_model(live_daily: pd.DataFrame, history: pd.DataFrame, location: Location) -> pd.DataFrame:
    model = joblib.load(LEGACY_MODEL_PATH)
    scaler = joblib.load(LEGACY_SCALER_PATH)
    features = legacy_feature_names(scaler)

    rows = []
    model_rows = []
    for _, day in live_daily.iterrows():
        u10, v10 = wind_components(float(day["wind_speed_api"]), float(day["wind_direction_api"]))
        moisture = nearest_monthly_moisture(history, location, int(day["date"].month))
        model_rows.append(
            {
                "lat": location.lat,
                "lon": location.lon,
                "sp": float(day["sp"]),
                "tcc": float(day["tcc"]),
                "u10": u10,
                "v10": v10,
                "t2m": float(day["t2m"]),
                # The existing scaler/model were trained before d2m was converted.
                "d2m": float(day["d2m"]) + 273.15,
                "lcc": float(day["lcc_percent"]) / 100.0,
                "viwve": float(moisture["viwve"]),
                "viwvn": float(moisture["viwvn"]),
            }
        )
        rows.append(
            {
                "location": location.name,
                "model_district": location.model_district,
                "lat": round(location.lat, 6),
                "lon": round(location.lon, 6),
                "date": day["date"].date().isoformat(),
                "t2m_mean_c": round(float(day["t2m"]), 3),
                "t2m_min_c": round(float(day["t2m_min"]), 3),
                "t2m_max_c": round(float(day["t2m_max"]), 3),
                "d2m_mean_c": round(float(day["d2m"]), 3),
                "sp_hpa": round(float(day["sp"]), 3),
                "cloud_cover_pct": round(float(day["tcc"]), 3),
                "api_precipitation_mm": round(float(day["api_precipitation_mm"]), 3),
            }
        )

    X = pd.DataFrame(model_rows).reindex(columns=features, fill_value=0)
    predictions = np.clip(model.predict(scaler.transform(X)), 0, None)
    output = pd.DataFrame(rows)
    output["predicted_rainfall_mm"] = np.round(predictions, 3)
    output["model_used"] = "legacy_model.pkl"
    return output


def predict_7day(district: str | None, lat: float | None, lon: float | None) -> pd.DataFrame:
    history = load_raw_history()
    location = resolve_location(history, district, lat, lon)
    live_daily = fetch_live_7day_weather(location.lat, location.lon)

    enhanced_ready = DRY_MODEL_PATH.exists() and MONSOON_MODEL_PATH.exists() and DISTRICT_ENCODER_PATH.exists()
    if enhanced_ready:
        forecast = predict_with_enhanced_models(live_daily, history, location)
        forecast["model_used"] = "enhanced_seasonal_xgboost"
        return forecast

    if not LEGACY_MODEL_PATH.exists() or not LEGACY_SCALER_PATH.exists():
        raise FileNotFoundError("No trained model found. Run: python3 merged_rainfall_prediction.py train")

    return predict_with_legacy_model(live_daily, history, location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run merged 7-day rainfall prediction.")
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser("predict", help="Predict rainfall for the next 7 days.")
    predict_parser.add_argument("--district", help="UP district name, for example: Agra")
    predict_parser.add_argument("--lat", type=float, help="Latitude for custom location")
    predict_parser.add_argument("--lon", type=float, help="Longitude for custom location")
    predict_parser.add_argument("--output", help="Optional CSV path for saving predictions")

    subparsers.add_parser("train", help="Train enhanced models using attached-notebook features.")

    parser.set_defaults(command="predict")
    parser.add_argument("--district", help=argparse.SUPPRESS)
    parser.add_argument("--lat", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lon", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        print("Please choose what you want to do.\n")
        print("Train improved models:")
        print("  python3 merged_rainfall_prediction.py train\n")
        print("Predict 7-day rainfall by district:")
        print("  python3 merged_rainfall_prediction.py predict --district Agra\n")
        print("Predict 7-day rainfall by latitude/longitude:")
        print("  python3 merged_rainfall_prediction.py predict --lat 27.1767 --lon 78.0081")
        return

    args = parse_args()

    if args.command == "train":
        train_enhanced_models()
        return

    forecast = predict_7day(args.district, args.lat, args.lon)
    print("\n7-day rainfall forecast")
    print(forecast.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        forecast.to_csv(output_path, index=False)
        print(f"\nSaved forecast to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
