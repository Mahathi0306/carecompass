"""The screen. Run with: streamlit run app.py"""
import os
import subprocess
import sys

import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

import engine as E

st.set_page_config(page_title="CareCompass", layout="wide")
st.title("CareCompass")
st.caption("Estimate cost, check schemes, find the right hospital. Indicative only, not medical advice.")

# On a fresh deployment the trained model file does not exist yet, so train it once.
if not os.path.exists("model/cost_model.joblib"):
    os.makedirs("model", exist_ok=True)
    with st.spinner("First run: training the cost model (takes a few seconds)..."):
        subprocess.run([sys.executable, "train.py"], check=True)

bundle = joblib.load("model/cost_model.joblib")
procs = pd.read_csv("data/procedures.csv")
schemes = E.load_schemes()
hospitals = E.load_hospitals()

if not bundle["used_real"]:
    st.warning("Demo mode: the model was trained on invented sample prices. Replace data/prices.csv with real data.")

# ---------------- inputs ----------------
with st.sidebar:
    st.header("Your details")
    procedure = st.selectbox("Procedure or test", procs["procedure"])
    room = st.selectbox("Room type (procedures only)", ["general", "semi", "private"])
    state = st.selectbox("State", ["Telangana", "Other"])
    cards = st.multiselect("Cards you hold", ["PMJAY card", "White ration card"])
    has_ins = st.checkbox("I have private insurance")
    sum_ins = st.number_input("Sum insured (Rs)", 0, 5000000, 300000, 50000) if has_ins else 0
    copay = st.slider("Co-pay %", 0, 50, 10) if has_ins else 0
    income = st.number_input("Family income per year (Rs)", 0, 10000000, 250000, 10000)
    lat = st.number_input("Your latitude", value=17.3850, format="%.4f")
    lon = st.number_input("Your longitude", value=78.4867, format="%.4f")
    max_km = st.slider("Maximum distance (km)", 5, 100, 40)
    mode = st.radio("Rank hospitals by", list(E.WEIGHTS.keys()))
    go = st.button("Get my plan", type="primary")

if go:
    st.session_state["show_plan"] = True      # remember the button was pressed
if not st.session_state.get("show_plan"):
    st.info("Fill the sidebar and press 'Get my plan'.")
    st.stop()

proc_row = procs[procs["procedure"] == procedure].iloc[0]
profile = {"state": state, "cards": cards}
insurance = {"sum_insured": sum_ins, "copay_pct": copay} if has_ins else None

eligible, rejected = E.match_schemes(profile, procedure, schemes)
ranked = E.rank_hospitals(hospitals, proc_row, lat, lon, max_km, room,
                          bundle, eligible, insurance, mode)

# ---------------- outputs ----------------
if ranked.empty:
    st.error("No hospital within this distance offers this procedure. Try a larger distance.")
    st.stop()

top = ranked.iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Estimated cost at top hospital", f"Rs {top.estimate:,.0f}")
c1.caption(f"Likely range: Rs {top.low:,.0f} to Rs {top.high:,.0f}")
c1.caption(f"Confidence: {E.confidence_label(bundle, procedure)}")
c2.metric("Estimated out-of-pocket", f"Rs {top.out_of_pocket:,.0f}")
c2.caption(top.case)
c3.metric("Financial burden", E.burden_band(top.out_of_pocket, income).split(" ")[0])
c3.caption(E.burden_band(top.out_of_pocket, income))

st.subheader("Schemes")
if eligible:
    for s in eligible:
        st.success(f"{s['name']}: documents to carry: " + ", ".join(s["documents"]))
else:
    st.write("No government scheme matched your details.")
with st.expander("Why other schemes did not match"):
    for r in rejected:
        st.write("- " + r)

st.subheader("Top hospitals")
show = ranked[["hospital", "type", "distance_km", "estimate", "out_of_pocket", "case", "score"]].copy()
show["estimate"] = show["estimate"].round(-2)
show["out_of_pocket"] = show["out_of_pocket"].round(-2)
show["score"] = show["score"].round(2)
st.dataframe(show, width='stretch')
st.caption(top.note)

best_cheap = ranked.loc[ranked["out_of_pocket"].idxmin()]
if best_cheap["out_of_pocket"] < top.out_of_pocket:
    st.info(f"Savings tip: {best_cheap.hospital} could save about Rs {top.out_of_pocket - best_cheap.out_of_pocket:,.0f}.")

# map
m = folium.Map(location=[lat, lon], zoom_start=11)
folium.Marker([lat, lon], tooltip="You", icon=folium.Icon(color="blue", icon="home")).add_to(m)
order = ranked["out_of_pocket"].rank(method="first")
for i, r in ranked.iterrows():
    color = "green" if r.out_of_pocket == 0 else ("orange" if order[i] <= 3 else "red")
    tag = " (accepts your scheme)" if r.scheme_ok else ""
    folium.Marker([r.lat, r.lon], tooltip=f"{r.hospital}{tag}: Rs {r.out_of_pocket:,.0f}",
                  icon=folium.Icon(color=color)).add_to(m)
st_folium(m, height=420, width=900, returned_objects=[])

# fair quote checker
st.subheader("Fair-quote checker")
pick = st.selectbox("Hospital", ranked["hospital"])
quote = st.number_input("Price the hospital quoted (Rs)", 0, 10000000, 0, 1000)
if quote > 0:
    row = ranked[ranked["hospital"] == pick].iloc[0]
    st.write(E.fair_quote(quote, row.low, row.high))
