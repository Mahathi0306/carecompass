"""Run with: py -m pytest
These tests use small INLINE examples, so they keep passing when you edit
your data files. They check the rules against hand-worked answers."""
import pandas as pd
import engine as E
 
SCHEMES = [
    {"id": "state_scheme", "name": "Test state scheme",
     "eligibility": {"states": ["Telangana"], "card_types": ["White ration card"]},
     "covered_procedures": ["Cataract surgery"], "documents": []},
    {"id": "pmjay", "name": "Test national scheme",
     "eligibility": {"states": ["All"], "card_types": ["PMJAY card"]},
     "covered_procedures": ["Cataract surgery"], "documents": []},
]
STATE_SCHEME = [SCHEMES[0]]
HOSP_ACCEPTS = {"schemes": ["state_scheme"]}
HOSP_NOT = {"schemes": []}
 
 
def test_haversine_zero_and_known_distance():
    assert E.haversine(17.385, 78.4867, 17.385, 78.4867) == 0
    # Hyderabad to Warangal is roughly 130 km in a straight line
    assert 110 < E.haversine(17.385, 78.4867, 17.9689, 79.5941) < 150
 
 
def test_scheme_match_needs_state_card_and_procedure():
    ok, _ = E.match_schemes({"state": "Telangana", "cards": ["White ration card"]}, "Cataract surgery", SCHEMES)
    assert [s["id"] for s in ok] == ["state_scheme"]
    no_card, why = E.match_schemes({"state": "Telangana", "cards": []}, "Cataract surgery", SCHEMES)
    assert no_card == [] and len(why) == 2
    wrong_state, _ = E.match_schemes({"state": "Other", "cards": ["White ration card"]}, "Cataract surgery", SCHEMES)
    assert wrong_state == []
    not_covered, _ = E.match_schemes({"state": "Telangana", "cards": ["White ration card"]}, "Hernia repair", SCHEMES)
    assert not_covered == []
 
 
def test_case1_scheme_at_empanelled_hospital_is_zero():
    amt, case, _, _ = E.out_of_pocket(100000, HOSP_ACCEPTS, STATE_SCHEME, None)
    assert amt == 0 and case.startswith("Case 1")
 
 
def test_case2_scheme_but_hospital_not_empanelled_pays_full():
    amt, case, _, _ = E.out_of_pocket(100000, HOSP_NOT, STATE_SCHEME, None)
    assert amt == 100000 and case.startswith("Case 2")
 
 
def test_case3_private_insurance_formula():
    # cost 100000, sum insured 300000, 10% co-pay -> covers 90000, patient pays 10000
    amt, case, _, _ = E.out_of_pocket(100000, HOSP_NOT, [], {"sum_insured": 300000, "copay_pct": 10})
    assert round(amt) == 10000 and case.startswith("Case 3")
    # cost above sum insured: 500000 cost, 300000 insured, 0% co-pay -> pays 200000
    amt, _, _, _ = E.out_of_pocket(500000, HOSP_NOT, [], {"sum_insured": 300000, "copay_pct": 0})
    assert round(amt) == 200000
 
 
def test_case4_no_coverage_pays_full():
    amt, case, _, _ = E.out_of_pocket(100000, HOSP_NOT, [], None)
    assert amt == 100000 and case.startswith("Case 4")
 
 
def test_fair_quote_bands():
    assert E.fair_quote(50, 100, 200).startswith("Below")
    assert E.fair_quote(150, 100, 200).startswith("Inside")
    assert E.fair_quote(250, 100, 200).startswith("Above")
 
 
def test_burden_band_thresholds():
    assert E.burden_band(5000, 100000).startswith("Green")
    assert E.burden_band(20000, 100000).startswith("Amber")
    assert E.burden_band(50000, 100000).startswith("Red")
 
 
def test_hospital_filters_procedure_and_distance():
    class FakeModel:                       # stands in for the trained model
        def predict(self, X): return [1.0] * len(X)
    bundle = {"model": FakeModel(), "p10": 0.8, "p90": 1.2, "real_counts": {}}
 
    def hosp(name, lat, lon, procs):
        return {"name": name, "lat": lat, "lon": lon, "type": "govt", "city_tier": 1,
                "accredited": 0, "rating": 4.0, "procedures": procs, "schemes": []}
    hospitals = pd.DataFrame([
        hosp("Offers it, near", 17.40, 78.48, ["Cataract surgery"]),
        hosp("Does not offer it", 17.41, 78.49, ["Hernia repair"]),
        hosp("Offers it, very far", 19.50, 80.50, ["Cataract surgery"]),
    ])
    proc = pd.Series({"procedure": "Cataract surgery", "category": "procedure", "reference_rate": 10000})
    out = E.rank_hospitals(hospitals, proc, 17.385, 78.4867, 50, "general",
                           bundle, [], None, "Cheapest")
    assert list(out["hospital"]) == ["Offers it, near"]
 























