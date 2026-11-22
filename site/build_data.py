import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))

from pm25_forecast import LAT, LON, POLLUTANTS, SPLIT, START, WEATHER, features

PKT = timezone(timedelta(hours=5))
BANDS = [
    {"name": "Good", "max": 10},
    {"name": "Fair", "max": 25},
    {"name": "Moderate", "max": 50},
    {"name": "Poor", "max": 75},
    {"name": "Very poor", "max": None},
]


def band(value):
    for i, b in enumerate(BANDS):
        if b["max"] is None or value < b["max"]:
            return i
    return len(BANDS) - 1


def get(url, params, key):
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    df = pd.DataFrame(r.json()[key])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def load(last_day):
    hourly = get("https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": LAT, "longitude": LON, "hourly": ",".join(POLLUTANTS),
        "start_date": START, "end_date": str(last_day), "timezone": "Asia/Karachi"}, "hourly")
    daily = hourly.sort_index().interpolate(limit=6).resample("D").mean().dropna()
    archive = get("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": LAT, "longitude": LON, "daily": ",".join(WEATHER),
        "start_date": "2022-08-01", "end_date": str(last_day), "timezone": "Asia/Karachi"}, "daily")
    recent = get("https://api.open-meteo.com/v1/forecast", {
        "latitude": LAT, "longitude": LON, "daily": ",".join(WEATHER),
        "past_days": 21, "forecast_days": 1, "timezone": "Asia/Karachi"}, "daily")
    weather = archive.combine_first(recent[recent.index <= pd.Timestamp(last_day)]).sort_index()
    return daily, weather


def model():
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=0)


def main():
    now = datetime.now(PKT)
    last_day = (now - timedelta(days=1)).date()
    daily, weather = load(last_day)
    X = features(daily, weather).dropna()
    origin = X.index[-1]
    out = {
        "updated": now.strftime("%Y-%m-%d %H:%M PKT"),
        "origin": str(origin.date()),
        "bands": BANDS,
        "latest": {"date": str(origin.date()), "pm25": round(float(daily.loc[origin, "pm2_5"]), 1),
                   "band": band(daily.loc[origin, "pm2_5"]),
                   "pollutants": {p: round(float(daily.loc[origin, p]), 1) for p in POLLUTANTS}},
        "history": [{"date": str(d.date()), "pm25": round(float(v), 1)} for d, v in daily["pm2_5"].loc[:origin].tail(120).items()],
        "forecast": [],
        "metrics": {},
        "training_days": int(len(X)),
        "first_day": str(X.index[0].date()),
    }
    for h in [1, 2, 3]:
        df = X.assign(target=daily["pm2_5"].shift(-h)).dropna()
        train = df.index < SPLIT - pd.Timedelta(days=h)
        test = df.index >= SPLIT
        cols = [c for c in df.columns if c != "target"]
        m = model().fit(df[cols][train], np.log1p(df["target"][train]))
        pred = np.expm1(m.predict(df[cols][test]))
        actual = df["target"][test]
        persistence = np.expm1(df["pm2_5"][test])
        mae, base = float(mean_absolute_error(actual, pred)), float(mean_absolute_error(actual, persistence))
        out["metrics"][f"h{h}"] = {
            "mae": round(mae, 2), "persistence_mae": round(base, 2), "r2": round(float(r2_score(actual, pred)), 2),
            "below_persistence_pct": round(100 * (1 - mae / base), 1), "test_days": int(test.sum()),
            "band_accuracy": round(100 * float(np.mean([band(a) == band(p) for a, p in zip(actual, pred)])), 1),
        }
        if h == 3:
            out["backtest"] = [{"date": str((d + pd.Timedelta(days=h)).date()), "actual": round(float(a), 1), "predicted": round(float(p), 1)}
                               for d, a, p in zip(actual.index, actual, pred)]
        final = model().fit(df[cols], np.log1p(df["target"]))
        value = float(np.expm1(final.predict(X[cols].iloc[[-1]]))[0])
        out["forecast"].append({"date": str((origin + pd.Timedelta(days=h)).date()), "horizon": h,
                                "pm25": round(value, 1), "band": band(value), "typical_error": round(mae, 1)})
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "dashboard.json"), "w") as f:
        json.dump(out, f)
    print(out["updated"], "origin", out["origin"], "latest", out["latest"]["pm25"], [(f["date"], f["pm25"]) for f in out["forecast"]])
    print(out["metrics"])


if __name__ == "__main__":
    main()
