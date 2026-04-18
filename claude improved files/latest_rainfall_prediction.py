"""
Latest rainfall prediction pipeline for the UP rainfall project.

This file combines:
- your current project assets: processed CSV with lat/lon, model.pkl, scaler.pkl
- the attached notebook's stronger feature engineering and seasonal models
- live 7-day weather values from Open-Meteo

Main usage:
    python3 latest_rainfall_prediction.py train
    python3 latest_rainfall_prediction.py predict --district Agra
    python3 latest_rainfall_prediction.py predict --lat 25.3176 --lon 82.9739

This version fetches recent observed rainfall for lag and rolling features,
and trains seven genuinely horizon-specific models (day_1..day_7): each one
learns to predict rainfall N days ahead from (a) that future day's own
weather and (b) antecedent-rain conditions known on the day the forecast is
made (see build_horizon_frame). Live prediction mirrors this exactly: the
rain_lag_*/rain_roll* features are anchored once from real observed data and
reused for all 7 days, instead of being recomputed from each day's own
prediction, which used to compound forecast error day over day.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
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
ARTIFACT_DIR = BASE_DIR / "artifacts_latest"

LEGACY_MODEL_PATH = BASE_DIR / "model.pkl"
LEGACY_SCALER_PATH = BASE_DIR / "scaler.pkl"

META_PATH = ARTIFACT_DIR / "rainfall_pipeline_meta.pkl"
BEST_REG_PATH = ARTIFACT_DIR / "best_regression_model.pkl"
BEST_CLS_PATH = ARTIFACT_DIR / "best_classification_model.pkl"
DRY_MODEL_PATH = ARTIFACT_DIR / "xgb_reg_dry.pkl"
MONSOON_MODEL_PATH = ARTIFACT_DIR / "xgb_reg_monsoon.pkl"
DISTRICT_ENCODER_PATH = ARTIFACT_DIR / "district_encoder.pkl"
DAY_MODEL_TEMPLATE = "xgb_day_{day}.pkl"

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


def get_with_retries(
    url: str,
    params: dict,
    *,
    timeout: int = 30,
    max_attempts: int = 3,
    backoff_seconds: float = 1.5,
) -> requests.Response:
    """GET with exponential backoff. Open-Meteo occasionally returns 429/5xx
    or times out; a single flaky call should not fail the whole forecast."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.RequestException(f"HTTP {response.status_code} from {url}")
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Request to {url} failed after {max_attempts} attempts: {last_exc}") from last_exc


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


def build_horizon_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Reshape the anchored per-date frame into a direct-forecast table for
    a specific horizon (1..7 days ahead).

    Previously every "day N" model was trained on the exact same
    (weather@T -> rainfall@T) rows regardless of N — the day-wise models
    were identical except for the XGBoost random_state, so there was no
    real horizon-specific signal despite the "day-wise" naming.

    This instead builds, per district and per row T:
      - weather/seasonal columns sourced from date T+horizon (what the
        live pipeline actually supplies: that forecast day's own weather)
      - antecedent-rain columns (rain_lag_*, rain_roll*) kept anchored at
        date T (what is actually known on the day the forecast is made —
        matches predict_with_enhanced_models, which no longer recomputes
        these recursively from its own predictions, see below)
      - target = rainfall_mm at T+horizon
    """
    anchored_cols = ["rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_lag_7", "rain_roll3", "rain_roll7"]
    shift_cols = [c for c in ENHANCED_FEATURES if c not in anchored_cols and c != "district_enc"]

    shifted = df.groupby("district")[shift_cols + ["rainfall_mm"]].shift(-horizon)
    out = df.copy()
    out[shift_cols] = shifted[shift_cols]
    out["target_rainfall_mm"] = shifted["rainfall_mm"]
    out = out.dropna(subset=shift_cols + ["target_rainfall_mm"]).reset_index(drop=True)
    return out


def train_enhanced_models() -> None:
    if XGBRegressor is None or XGBClassifier is None:
        raise RuntimeError("xgboost is required for enhanced training.") from XGBOOST_IMPORT_ERROR

    ARTIFACT_DIR.mkdir(exist_ok=True)
    base_df, encoder = build_enhanced_training_frame()

    def time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr = frame[frame["year"] <= 2022].copy()
        te = frame[frame["year"] >= 2023].copy()
        if tr.empty or te.empty:
            cutoff = int(frame["year"].quantile(0.8))
            tr = frame[frame["year"] <= cutoff].copy()
            te = frame[frame["year"] > cutoff].copy()
        return tr, te

    # Outlier cap derived once from the horizon-1 TRAIN split only, then
    # reused everywhere (train/val/test, every horizon) so the cap never
    # leaks information from held-out rows or from other horizons' shifted
    # targets.
    h1_train, _ = time_split(build_horizon_frame(base_df, 1))
    rainfall_cap = float(h1_train["target_rainfall_mm"].quantile(0.99))

    xgb_acc = None
    day_metrics = {}
    total_train_rows = 0
    total_test_rows = 0

    for forecast_day in range(1, 8):
        horizon_df = build_horizon_frame(base_df, forecast_day)
        horizon_df["target_rainfall_mm"] = horizon_df["target_rainfall_mm"].clip(upper=rainfall_cap)
        horizon_df["target_rain_occurred"] = (horizon_df["target_rainfall_mm"] > 0.1).astype(int)

        train, test = time_split(horizon_df)
        train = train.sort_values("date")
        val_cut = int(len(train) * 0.85)
        fit_part = train.iloc[:val_cut]
        val_part = train.iloc[val_cut:]

        X_fit = fit_part[ENHANCED_FEATURES]
        X_val = val_part[ENHANCED_FEATURES]
        X_test = test[ENHANCED_FEATURES]

        model = XGBRegressor(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            gamma=1.0,
            reg_alpha=0.2,
            reg_lambda=2.0,
            n_jobs=-1,
            random_state=42 + forecast_day,
            verbosity=0,
            early_stopping_rounds=30,
            eval_metric="mae",
        )
        model.fit(
            X_fit,
            np.log1p(fit_part["target_rainfall_mm"]),
            eval_set=[(X_val, np.log1p(val_part["target_rainfall_mm"]))],
            verbose=False,
        )

        pred = np.clip(np.expm1(model.predict(X_test)), 0, None)
        actual = test["target_rainfall_mm"]
        day_metrics[forecast_day] = {
            "mae": float(mean_absolute_error(actual, pred)),
            "r2": float(r2_score(actual, pred)),
            "best_iteration": int(getattr(model, "best_iteration", model.n_estimators)),
            "train_rows": int(len(fit_part)),
            "val_rows": int(len(val_part)),
            "test_rows": int(len(test)),
        }
        joblib.dump(model, ARTIFACT_DIR / DAY_MODEL_TEMPLATE.format(day=forecast_day))
        total_train_rows += len(fit_part)
        total_test_rows += len(test)

        # Train the "will it rain" classifier once, on the horizon-1 table
        # (predicting tomorrow's rain occurrence from today's conditions —
        # the most defensible framing for a reused daily rain-probability
        # estimate). Previously this was trained on same-date rows only.
        if forecast_day == 1:
            y_fit_cls = fit_part["target_rain_occurred"]
            y_val_cls = val_part["target_rain_occurred"]
            y_test_cls = test["target_rain_occurred"]
            xgb_cls = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_lambda=2.0,
                n_jobs=-1,
                random_state=42,
                verbosity=0,
                eval_metric="logloss",
                early_stopping_rounds=30,
            )
            xgb_cls.fit(X_fit, y_fit_cls, eval_set=[(X_val, y_val_cls)], verbose=False)
            xgb_acc = float(accuracy_score(y_test_cls, xgb_cls.predict(X_test)))

    meta = {
        "features": ENHANCED_FEATURES,
        "rainfall_cap": rainfall_cap,
        "train_rows": total_train_rows,
        "test_rows": total_test_rows,
        "xgb_cls_accuracy": xgb_acc,
        "day_model_metrics": day_metrics,
    }

    joblib.dump(meta, META_PATH)
    joblib.dump(encoder, DISTRICT_ENCODER_PATH)
    joblib.dump(xgb_cls, BEST_CLS_PATH)

    print("Latest day-wise models trained and saved in:", ARTIFACT_DIR)
    print(f"Rain classifier accuracy={meta['xgb_cls_accuracy'] * 100:.2f}%")
    for forecast_day, metrics in day_metrics.items():
        print(
            f"Day {forecast_day}: "
            f"MAE={metrics['mae']:.2f} mm, R2={metrics['r2']:.4f}, "
            f"train={metrics['train_rows']}, test={metrics['test_rows']}"
        )


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
    if len(series) < 14:
        series = ([0.0] * (14 - len(series))) + series
    return [float(x) for x in series]


def fetch_recent_14day_rainfall(lat: float, lon: float) -> list[float]:
    """Fetch recent observed rainfall for lag features used by the live forecast."""
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=13)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "precipitation_sum",
        "timezone": "auto",
    }

    try:
        response = get_with_retries(url, params)
        payload = response.json()
        rain = payload["daily"]["precipitation_sum"]
    except (KeyError, TypeError, RuntimeError, requests.RequestException) as exc:
        raise RuntimeError("Could not fetch recent rainfall history from Open-Meteo Archive.") from exc

    rain = [0.0 if value is None else float(value) for value in rain]
    if len(rain) < 14:
        rain = ([0.0] * (14 - len(rain))) + rain
    return rain[-14:]


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
        response = get_with_retries(url, params)
    except (RuntimeError, requests.RequestException) as exc:
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
    classifier = joblib.load(BEST_CLS_PATH) if BEST_CLS_PATH.exists() else None

    try:
        rain_memory = fetch_recent_14day_rainfall(location.lat, location.lon)
        rain_memory_source = "live_archive_api"
    except RuntimeError:
        rain_memory = latest_rain_memory(history, location.model_district)
        rain_memory_source = "historical_dataset_fallback"

    # Anchor the lag/rolling-rain features ONCE, from real observed data as
    # of "today" (the day the forecast is made). Every horizon model
    # (day_1..day_7) was trained this way in build_horizon_frame: antecedent
    # rain conditions come from the anchor date, only the weather columns
    # move forward per horizon. Previously these values were recomputed
    # each iteration by appending each day's own PREDICTION back into
    # rain_memory, so day 2 onward was scored on a feature distribution
    # (predicted lags) that the models never saw during training, and any
    # early over/under-prediction compounded into every later day.
    anchored_lags = {
        "rain_lag_1": rain_memory[-1],
        "rain_lag_2": rain_memory[-2],
        "rain_lag_3": rain_memory[-3],
        "rain_lag_7": rain_memory[-7],
        "rain_roll3": float(np.mean(rain_memory[-3:])),
        "rain_roll7": float(np.mean(rain_memory[-7:])),
    }

    rows = []

    for forecast_day, (_, day) in enumerate(live_daily.iterrows(), start=1):
        one_day = pd.DataFrame([day])
        features_df, display_df = build_live_enhanced_features(one_day, history, location, encoder)

        for col, value in anchored_lags.items():
            features_df.loc[0, col] = value

        X = features_df.reindex(columns=ENHANCED_FEATURES, fill_value=0)
        model_path = ARTIFACT_DIR / DAY_MODEL_TEMPLATE.format(day=forecast_day)
        model = joblib.load(model_path)
        predicted = float(np.expm1(model.predict(X)[0]))
        predicted = max(predicted, 0.0)

        display = display_df.iloc[0].to_dict()
        display["forecast_day"] = forecast_day
        display["predicted_rainfall_mm"] = round(predicted, 3)
        display["lag_rainfall_source"] = rain_memory_source
        if classifier is not None and hasattr(classifier, "predict_proba"):
            display["rain_probability_pct"] = round(float(classifier.predict_proba(X)[0, 1]) * 100, 2)

        rows.append(display)

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

    day_models_ready = all((ARTIFACT_DIR / DAY_MODEL_TEMPLATE.format(day=day)).exists() for day in range(1, 8))
    enhanced_ready = day_models_ready and DISTRICT_ENCODER_PATH.exists()
    if enhanced_ready:
        forecast = predict_with_enhanced_models(live_daily, history, location)
        forecast["model_used"] = "latest_daywise_xgboost"
        return forecast

    if not LEGACY_MODEL_PATH.exists() or not LEGACY_SCALER_PATH.exists():
        raise FileNotFoundError("No trained model found. Run: python3 latest_rainfall_prediction.py train")

    return predict_with_legacy_model(live_daily, history, location)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run latest day-wise 7-day rainfall prediction.")
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser("predict", help="Predict rainfall for the next 7 days.")
    predict_parser.add_argument("--district", help="UP district name, for example: Agra")
    predict_parser.add_argument("--lat", type=float, help="Latitude for custom location")
    predict_parser.add_argument("--lon", type=float, help="Longitude for custom location")
    predict_parser.add_argument("--output", help="Optional CSV path for saving predictions")

    subparsers.add_parser("train", help="Train latest day-wise models using enhanced features.")

    parser.set_defaults(command="predict")
    parser.add_argument("--district", help=argparse.SUPPRESS)
    parser.add_argument("--lat", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--lon", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    if len(sys.argv) == 1:
        print("Please choose what you want to do.\n")
        print("Train latest improved models:")
        print("  python3 latest_rainfall_prediction.py train\n")
        print("Predict 7-day rainfall by district:")
        print("  python3 latest_rainfall_prediction.py predict --district Agra\n")
        print("Predict 7-day rainfall by latitude/longitude:")
        print("  python3 latest_rainfall_prediction.py predict --lat 25.3176 --lon 82.9739")
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
