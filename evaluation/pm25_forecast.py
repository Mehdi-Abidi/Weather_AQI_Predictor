import json
import os

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LAT, LON = 24.8607, 67.0011
START, END = "2022-08-05", "2026-11-20"
SPLIT = pd.Timestamp("2025-11-01")
POLLUTANTS = ["pm2_5", "pm10", "nitrogen_dioxide", "ozone", "carbon_monoxide", "sulphur_dioxide"]
WEATHER = ["temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max",
           "wind_direction_10m_dominant", "precipitation_sum", "surface_pressure_mean"]
PM25_BANDS = [10, 25, 50, 75]


def cached(name, url, params, key):
    path = os.path.join(DATA, name)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["time"])
    os.makedirs(DATA, exist_ok=True)
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json()[key])
    df["time"] = pd.to_datetime(df["time"])
    df.to_csv(path, index=False)
    return df


def air_quality():
    return cached("karachi_air_quality_hourly.csv", "https://air-quality-api.open-meteo.com/v1/air-quality", {
        "latitude": LAT, "longitude": LON, "hourly": ",".join(POLLUTANTS),
        "start_date": START, "end_date": END, "timezone": "Asia/Karachi"}, "hourly")


def weather():
    return cached("karachi_weather_daily.csv", "https://archive-api.open-meteo.com/v1/archive", {
        "latitude": LAT, "longitude": LON, "daily": ",".join(WEATHER),
        "start_date": "2022-08-01", "end_date": END, "timezone": "Asia/Karachi"}, "daily").set_index("time")


def daily_pollutants():
    hourly = air_quality().sort_values("time").set_index("time").interpolate(limit=6)
    return hourly.resample("D").mean().dropna()


def features(d, wx):
    X = pd.DataFrame(index=d.index)
    for col in POLLUTANTS:
        X[col] = np.log1p(d[col])
    for lag in range(1, 8):
        X[f"pm25_lag{lag}"] = np.log1p(d["pm2_5"].shift(lag))
    for w in [3, 7, 14, 30]:
        X[f"pm25_mean{w}"] = np.log1p(d["pm2_5"].rolling(w).mean())
    X["pm10_mean7"] = np.log1p(d["pm10"].rolling(7).mean())
    doy = d.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    w = wx.copy()
    w["wind_u"] = w["wind_speed_10m_max"] * np.sin(np.radians(w["wind_direction_10m_dominant"]))
    w["wind_v"] = w["wind_speed_10m_max"] * np.cos(np.radians(w["wind_direction_10m_dominant"]))
    return X.join(w.drop(columns=["wind_direction_10m_dominant"]).add_prefix("wx_"), how="left")


def metrics(y, p):
    return {"mae": round(float(mean_absolute_error(y, p)), 2),
            "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 2),
            "r2": round(float(r2_score(y, p)), 3)}


def run(horizon):
    d = daily_pollutants()
    df = features(d, weather()).assign(target=d["pm2_5"].shift(-horizon)).dropna()
    train = df.index < SPLIT - pd.Timedelta(days=horizon)
    test = df.index >= SPLIT
    Xtr, Xte = df.drop(columns=["target"])[train], df.drop(columns=["target"])[test]
    ytr, yte = df["target"][train], df["target"][test]
    res = {"horizon_days": horizon, "train_days": int(train.sum()), "test_days": int(test.sum()),
           "test_period": [str(df.index[test][0].date()), str(df.index[test][-1].date())],
           "test_mean_pm25": round(float(yte.mean()), 1)}
    baselines = {
        "persistence": np.expm1(Xte["pm2_5"]),
        "mean_last_7_days": np.expm1(Xte["pm25_mean7"]),
        "monthly_climatology": yte.index.month.map(ytr.groupby(ytr.index.month).mean()).to_numpy(),
    }
    for name, pred in baselines.items():
        res[name] = metrics(yte, pred)
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "random_forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, n_jobs=-1, random_state=0),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, random_state=0),
    }
    preds = {}
    for name, model in models.items():
        model.fit(Xtr, np.log1p(ytr))
        preds[name] = np.expm1(model.predict(Xte))
        res[name] = metrics(yte, preds[name])
    for name in list(models) + ["mean_last_7_days", "monthly_climatology"]:
        res[name]["mae_below_persistence_pct"] = round(100 * (1 - res[name]["mae"] / res["persistence"]["mae"]), 1)
    best = min(models, key=lambda n: res[n]["mae"])
    res["best_model"] = best
    band_true = np.digitize(yte.to_numpy(), PM25_BANDS)
    res["pm25_band_accuracy"] = {
        best: round(float(np.mean(np.digitize(preds[best], PM25_BANDS) == band_true)), 3),
        "persistence": round(float(np.mean(np.digitize(baselines["persistence"].to_numpy(), PM25_BANDS) == band_true)), 3),
    }
    return res


if __name__ == "__main__":
    report = {f"horizon_{h}_day": run(h) for h in [1, 3]}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(report, f, indent=1)
    for name, res in report.items():
        print(name)
        for k, v in res.items():
            print("  ", k, v)
