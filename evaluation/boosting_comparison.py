import json
import os
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from pm25_forecast import HERE, SPLIT, daily_pollutants, features, metrics, weather

warnings.filterwarnings("ignore")

GRIDS = {
    "xgboost": (XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=1),
                {"n_estimators": [300, 600], "max_depth": [2, 4], "learning_rate": [0.02, 0.05], "min_child_weight": [1, 5]}),
    "lightgbm": (LGBMRegressor(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, n_jobs=1, verbose=-1),
                 {"n_estimators": [300, 600], "num_leaves": [4, 12], "learning_rate": [0.02, 0.05], "min_child_samples": [10, 30]}),
    "random_forest": (RandomForestRegressor(random_state=0, n_jobs=1),
                      {"n_estimators": [500], "min_samples_leaf": [3, 5, 10], "max_features": [0.33, 1.0]}),
}


def run(horizon):
    d = daily_pollutants()
    df = features(d, weather()).assign(target=d["pm2_5"].shift(-horizon)).dropna()
    train = df.index < SPLIT - pd.Timedelta(days=horizon)
    test = df.index >= SPLIT
    Xtr, Xte = df.drop(columns=["target"])[train], df.drop(columns=["target"])[test]
    ytr, yte = df["target"][train], df["target"][test]
    folds = TimeSeriesSplit(n_splits=5, gap=horizon)
    res = {"persistence": metrics(yte, np.expm1(Xte["pm2_5"]))}
    for name, (model, grid) in GRIDS.items():
        search = GridSearchCV(model, grid, cv=folds, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(Xtr, np.log1p(ytr))
        res[name] = dict(metrics(yte, np.expm1(search.predict(Xte))), chosen=search.best_params_)
    return res


if __name__ == "__main__":
    report = {f"horizon_{h}_day": run(h) for h in [1, 3]}
    with open(os.path.join(HERE, "boosting_results.json"), "w") as f:
        json.dump(report, f, indent=1)
    for name, res in report.items():
        print(name)
        for k, v in res.items():
            print("  ", k, v)
