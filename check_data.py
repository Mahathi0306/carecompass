"""Run after editing any data file:  py check_data.py
It catches typos and mismatches BEFORE they silently break the app."""
import difflib
import json
import pandas as pd
 
problems = []
procs = pd.read_csv("data/procedures.csv")
prices = pd.read_csv("data/prices.csv")
hosp = pd.read_csv("data/hospitals.csv")
schemes = json.load(open("data/schemes.json", encoding="utf-8"))
 
def numeric_check(df, col, file):
    """Finds cells that are not plain numbers (like '45,000' or 'Rs 500') and reports the row."""
    bad = pd.to_numeric(df[col], errors="coerce").isna()
    for i in df.index[bad]:
        val = str(df.loc[i, col])
        if col in ("lat", "lon") and "," in val:
            problems.append(f"{file}: row {i + 2}, column {col} holds BOTH numbers ('{val}'). "
                            "Put the first number in lat and the second in lon (two cells).")
        else:
            problems.append(f"{file}: row {i + 2}, column {col} is not a plain number: '{val}' "
                            "(use digits only, no commas, no Rs, no spaces)")
    df[col] = pd.to_numeric(df[col], errors="coerce")
 
numeric_check(procs, "reference_rate", "procedures.csv")
for c in ["reference_rate", "actual_price", "city_tier", "accredited"]:
    if c in prices.columns:
        numeric_check(prices, c, "prices.csv")
for c in ["lat", "lon", "city_tier", "accredited", "rating"]:
    if c in hosp.columns:
        numeric_check(hosp, c, "hospitals.csv")
 
def hint(name):
    close = difflib.get_close_matches(name.strip().lower(),
                                      [n.lower() for n in procs["procedure"]], n=1, cutoff=0.6)
    if close:
        real = [n for n in procs["procedure"] if n.lower() == close[0]][0]
        return f" -> did you mean '{real}'?"
    return " -> not in procedures.csv (delete it from this list, or add it to procedures.csv)"
 
# stray spaces in names
for col in ["procedure"]:
    for df, f in [(procs, "procedures.csv"), (prices, "prices.csv")]:
        if col in df.columns:
            sp = df[col].astype(str) != df[col].astype(str).str.strip()
            for i in df.index[sp]:
                problems.append(f"{f}: row {i + 2}, '{df.loc[i, col]}' has a space at the start or end")
 
names = set(procs["procedure"])
scheme_ids = {s["id"] for s in schemes}
 
# procedures.csv
if not set(procs["category"]) <= {"procedure", "diagnostic"}:
    problems.append("procedures.csv: category must be 'procedure' or 'diagnostic'")
if (procs["reference_rate"].dropna() <= 0).any():
    problems.append("procedures.csv: reference_rate must be positive")
 
# prices.csv
for col in ["procedure", "reference_rate", "city_tier", "hospital_type", "accredited",
            "room_type", "actual_price", "source", "data_type"]:
    if col not in prices.columns:
        problems.append(f"prices.csv: missing column {col}")
if not problems:
    bad = set(prices["procedure"]) - names
    if bad: problems.append(f"prices.csv: procedure names not in procedures.csv: {sorted(bad)}")
    if not set(prices["hospital_type"]) <= {"govt", "private", "trust"}:
        problems.append("prices.csv: hospital_type must be govt, private or trust")
    if not set(prices["city_tier"]) <= {1, 2, 3}:
        problems.append("prices.csv: city_tier must be 1, 2 or 3")
    if not set(prices["accredited"]) <= {0, 1}:
        problems.append("prices.csv: accredited must be 0 or 1")
    if not set(prices["room_type"]) <= {"general", "semi", "private", "none"}:
        problems.append("prices.csv: room_type must be general, semi, private or none")
    real = prices[prices["data_type"] != "synthetic"]
    if real["source"].isna().any() or (real["source"].astype(str).str.strip() == "").any():
        problems.append("prices.csv: every real row needs a source")
    ref = prices.merge(procs, on="procedure", suffixes=("", "_official"))
    if (ref["reference_rate"] != ref["reference_rate_official"]).any():
        problems.append("prices.csv: reference_rate differs from procedures.csv for some rows")
 
# hospitals.csv
for _, h in hosp.iterrows():
    for p in str(h["procedures"]).split(";"):
        p = p.strip()
        if p and p not in names:
            problems.append(f"hospitals.csv: '{h['name']}' lists '{p}'" + hint(p))
    for s in str(h["schemes"]).split(";"):
        s = s.strip()
        if s and s != "nan" and s not in scheme_ids:
            problems.append(f"hospitals.csv: '{h['name']}' lists unknown scheme id '{s}'")
 
# schemes.json
for s in schemes:
    for p in s["covered_procedures"]:
        if p not in names:
            problems.append(f"schemes.json: {s['id']} covers '{p}'" + hint(p))
    if "ADD" in s.get("source", "") or "ADD" in s.get("last_checked", ""):
        problems.append(f"schemes.json: {s['id']} still has placeholder source or date")
 
# summary
print(f"{len(procs)} procedures, {len(prices)} price rows, {len(hosp)} hospitals, {len(schemes)} schemes")
real = prices[prices["data_type"] != "synthetic"]
print(f"Real price rows: {len(real)} (need 20+ for honest testing)")
print("Real rows per procedure:")
print(real.groupby("procedure").size().reindex(procs["procedure"]).fillna(0).astype(int).to_string())
print()
if problems:
    print("PROBLEMS TO FIX:")
    for p in problems: print(" -", p)
else:
    print("All checks passed.")
 