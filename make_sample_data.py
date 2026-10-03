"""Creates DEMO price data so you can run the whole pipeline today.
These rows are INVENTED (data_type = synthetic). Replace them with real,
sourced rows in data/prices.csv as soon as you collect them."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
procs = pd.read_csv("data/procedures.csv")

type_base = {"govt": 1.0, "trust": 1.25, "private": 1.7}
tier_mult = {1: 1.15, 2: 1.0, 3: 0.9}
room_mult = {"general": 1.0, "semi": 1.1, "private": 1.25, "none": 1.0}

rows = []
for _ in range(400):
    p = procs.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
    tier = int(rng.choice([1, 2, 3]))
    htype = str(rng.choice(["govt", "trust", "private"]))
    acc = int(rng.integers(0, 2))
    room = "none" if p.category == "diagnostic" else str(rng.choice(["general", "semi", "private"]))
    ratio = type_base[htype] * tier_mult[tier] * room_mult[room] * (1 + 0.12 * acc)
    ratio *= float(np.exp(rng.normal(0, 0.12)))
    rows.append({
        "procedure": p.procedure, "reference_rate": p.reference_rate,
        "city_tier": tier, "hospital_type": htype, "accredited": acc,
        "room_type": room, "age_group": str(rng.choice(["18-40", "41-60", "60+"])),
        "actual_price": round(p.reference_rate * ratio, -1),
        "source": "invented demo row", "data_type": "synthetic",
    })
pd.DataFrame(rows).to_csv("data/prices.csv", index=False)
print("Wrote data/prices.csv with", len(rows), "demo rows")
