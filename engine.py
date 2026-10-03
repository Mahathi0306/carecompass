"""All the logic of CareCompass (no screens here, only thinking)."""
import json
import math
import numpy as np
import pandas as pd

CAT_FEATURES = ["city_tier", "hospital_type", "room_type", "category"]
FEATURES = CAT_FEATURES + ["accredited"]


class BaselineGroupMean:
    """The simple guess the ML models must beat:
    average ratio for the same hospital type and city tier."""
    def __init__(self, keys=("hospital_type", "city_tier")):
        self.keys = list(keys)

    def fit(self, X, y):
        df = X[self.keys].copy()
        df["y"] = np.asarray(y)
        self.table_ = df.groupby(self.keys)["y"].mean()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.array([self.table_.get(tuple(r), self.global_)
                         for r in X[self.keys].itertuples(index=False)])


# ---------- distance ----------
def haversine(lat1, lon1, lat2, lon2):
    """Straight-line distance in km between two points on Earth."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------- loading ----------
def load_schemes(path="data/schemes.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_hospitals(path="data/hospitals.csv"):
    df = pd.read_csv(path)
    df["procedures"] = df["procedures"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()])
    df["schemes"] = df["schemes"].fillna("").apply(
        lambda s: [x.strip() for x in s.split(";") if x.strip()])
    return df


# ---------- cost engine ----------
def estimate_cost(bundle, proc_row, hosp, room_type):
    """Returns (estimate, low, high) in rupees for one hospital."""
    row = pd.DataFrame([{
        "city_tier": int(hosp["city_tier"]),
        "hospital_type": hosp["type"],
        "room_type": room_type if proc_row["category"] == "procedure" else "none",
        "category": proc_row["category"],
        "accredited": int(hosp["accredited"]),
    }])
    ratio = max(float(bundle["model"].predict(row)[0]), 0.3)
    est = ratio * float(proc_row["reference_rate"])
    return est, est * bundle["p10"], est * bundle["p90"]


def confidence_label(bundle, procedure):
    n = bundle["real_counts"].get(procedure, 0)
    if n >= 15:
        return "High"
    if n >= 5:
        return "Medium"
    return "Low (little or no real data, treat as indicative)"


# ---------- scheme engine (rules, not ML) ----------
def match_schemes(profile, procedure, schemes):
    """Returns (eligible_schemes, reasons_for_rejected)."""
    eligible, rejected = [], []
    for s in schemes:
        e = s["eligibility"]
        state_ok = "All" in e["states"] or profile["state"] in e["states"]
        card_ok = any(c in profile["cards"] for c in e["card_types"])
        covered = procedure in s["covered_procedures"]
        if state_ok and card_ok and covered:
            eligible.append(s)
        else:
            why = []
            if not state_ok: why.append("state does not match")
            if not card_ok: why.append("required card not held")
            if not covered: why.append("procedure not covered")
            rejected.append(f"{s['name']}: " + ", ".join(why))
    return eligible, rejected


def out_of_pocket(cost, hosp, eligible, insurance):
    """The four cases. Returns (amount, case_name, note, scheme_used)."""
    accepted = [s for s in eligible if s["id"] in hosp["schemes"]]
    if accepted:
        s = accepted[0]
        return (0.0, "Case 1: scheme + empanelled hospital",
                f"Covered under {s['name']}. Extras outside the package may still apply.", s)
    if insurance and insurance["sum_insured"] > 0:
        cov = min(cost, insurance["sum_insured"]) * (1 - insurance["copay_pct"] / 100)
        return (max(cost - cov, 0.0), "Case 3: private insurance",
                "Waiting periods, pre-existing conditions, room-rent caps and non-payable items can reduce this.", None)
    if eligible:
        return (cost, "Case 2: scheme not usable here",
                "You are eligible for a scheme, but this hospital is not empanelled for it.", None)
    return (cost, "Case 4: no coverage", "No scheme or insurance applies.", None)


# ---------- navigator ----------
WEIGHTS = {
    "Cheapest":     dict(cost=0.5, dist=0.2, quality=0.1, scheme=0.2),
    "Nearest":      dict(cost=0.2, dist=0.5, quality=0.1, scheme=0.2),
    "Best quality": dict(cost=0.2, dist=0.1, quality=0.5, scheme=0.2),
}


def _scale_low_is_good(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    return 1 - (series - lo) / (hi - lo)


def rank_hospitals(hospitals, proc_row, user_lat, user_lon, max_km, room_type,
                   bundle, eligible, insurance, mode):
    rows = []
    for _, h in hospitals.iterrows():
        # Stage 1: hard filters
        if proc_row["procedure"] not in h["procedures"]:
            continue
        d = haversine(user_lat, user_lon, h["lat"], h["lon"])
        if d > max_km:
            continue
        # cost + coverage at THIS hospital
        est, low, high = estimate_cost(bundle, proc_row, h, room_type)
        oop, case, note, scheme = out_of_pocket(est, h, eligible, insurance)
        rows.append({"hospital": h["name"], "lat": h["lat"], "lon": h["lon"],
                     "type": h["type"], "distance_km": round(d, 1),
                     "estimate": est, "low": low, "high": high,
                     "out_of_pocket": oop, "case": case, "note": note,
                     "rating": h["rating"], "scheme_ok": 1 if scheme else 0})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Stage 2: scoring (every factor scaled 0 to 1)
    w = WEIGHTS[mode]
    df["score"] = (w["cost"] * _scale_low_is_good(df["out_of_pocket"])
                   + w["dist"] * _scale_low_is_good(df["distance_km"])
                   + w["quality"] * (df["rating"] / 5.0)
                   + w["scheme"] * df["scheme_ok"])
    return df.sort_values("score", ascending=False).head(5).reset_index(drop=True)


# ---------- extra tools ----------
def fair_quote(quote, low, high):
    if quote < low: return "Below the usual range"
    if quote > high: return "Above the usual range (ask for an itemised bill)"
    return "Inside the usual range"


def burden_band(oop, annual_income):
    if annual_income <= 0: return "Unknown"
    r = oop / annual_income
    if r < 0.10: return "Green (under 10% of yearly income)"
    if r < 0.30: return "Amber (10% to 30% of yearly income)"
    return "Red (over 30% of yearly income)"