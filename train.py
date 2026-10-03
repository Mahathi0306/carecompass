"""Run once: python train.py
Compares 4 models by repeated 5-fold cross-validation, picks the best,
measures its typical error to build the range, and saves everything."""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from engine import BaselineGroupMean, CAT_FEATURES, FEATURES

os.makedirs("model", exist_ok=True)
prices = pd.read_csv("data/prices.csv")
procs = pd.read_csv("data/procedures.csv")[["procedure", "category"]]
df = prices.merge(procs, on="procedure", how="left").reset_index(drop=True)

df["ratio"] = df["actual_price"] / df["reference_rate"]   # what we predict
X = df[FEATURES]
y = df["ratio"]

# Honesty rule: test only on REAL rows
real_mask = (df["data_type"] != "synthetic").to_numpy()
used_real = bool(real_mask.sum() >= 20)
if not used_real:
    print("WARNING: fewer than 20 real rows. Scores below are DEMO ONLY.")
    real_mask = np.ones(len(df), dtype=bool)


def make_pipe(model):
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)],
        remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


candidates = {
    "Baseline": lambda: BaselineGroupMean(),
    "Ridge": lambda: make_pipe(Ridge(alpha=1.0)),
    "Random Forest": lambda: make_pipe(RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0)),
    "Gradient Boosting": lambda: make_pipe(GradientBoostingRegressor(random_state=0)),
}

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
results, all_err_ratios = [], {}
for name, build in candidates.items():
    abs_err, pct_err, err_ratio = [], [], []
    for tr, te in rkf.split(df):
        te = te[real_mask[te]]
        if len(te) == 0:
            continue
        m = build().fit(X.iloc[tr], y.iloc[tr])
        pred_ratio = np.clip(m.predict(X.iloc[te]), 0.3, None)
        ref = df["reference_rate"].iloc[te].to_numpy()
        actual = df["actual_price"].iloc[te].to_numpy()
        pred_price = pred_ratio * ref
        abs_err += list(np.abs(actual - pred_price))
        pct_err += list(np.abs(actual - pred_price) / actual * 100)
        err_ratio += list(actual / pred_price)
    results.append({"model": name, "MAE_rupees": round(float(np.mean(abs_err))),
                    "MAPE_percent": round(float(np.mean(pct_err)), 1)})
    all_err_ratios[name] = err_ratio

res = pd.DataFrame(results).sort_values("MAE_rupees")
print(res.to_string(index=False))
res.to_csv("model/results.csv", index=False)

best = res.iloc[0]["model"]
p10, p90 = np.percentile(all_err_ratios[best], [10, 90])
inside = np.mean((np.array(all_err_ratios[best]) >= p10) & (np.array(all_err_ratios[best]) <= p90)) * 100
print(f"\nBest model: {best}")
print(f"Range multipliers: {p10:.2f} to {p90:.2f}  (covers {inside:.0f}% of held-out prices)")

final = candidates[best]().fit(X, y)          # refit on all data
real_counts = (df[df["data_type"] != "synthetic"].groupby("procedure").size().to_dict())
joblib.dump({"model": final, "name": best, "p10": float(p10), "p90": float(p90),
             "real_counts": real_counts, "used_real": used_real}, "model/cost_model.joblib")
print("Saved model/cost_model.joblib")
