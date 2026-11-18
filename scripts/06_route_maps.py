"""Folium maps of student-test trips run through the deployed inference pipeline:
raw GPS, road network, reference route, classifier-only route and classifier+HMM route."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from pathsnap.data_io import ROOT, load_trip
from pathsnap.core import Trip
from pathsnap.pipeline import PathSnapPredictor
from pathsnap.maps import build_map

MAPS = ROOT / "results" / "maps"; MAPS.mkdir(parents=True, exist_ok=True)
per = pd.read_csv(ROOT / "results" / "hmm_per_trip_student_test.csv", dtype={"trip": str})
h = per[per.variant == "Classifier + HMM/Viterbi"].sort_values("top1")
picks = [h.trip.iloc[-1], h.trip.iloc[len(h) // 2], h.trip.iloc[0]]
pred = PathSnapPredictor()
for tid in picks:
    tid = str(tid).zfill(8)
    out = pred.predict(load_trip("student_test", tid, with_route=False), tid)
    ref = Trip(load_trip("student_test", tid), tid).route      # reference only for display
    m = build_map(out["trip"], ref_arcs=ref,
                  pred_layers=[("Classifier-only route", out["clf_arcs"]), ("Predicted route (classifier + HMM)", out["arcs"])],
                  title=f"Student-test trip {tid}")
    m.save(str(MAPS / f"route_student_test_{tid}.html"))
    # detail view: 300-observation window where classifier-only and HMM disagree most
    diff = (out["clf_arcs"] != out["arcs"]).astype(float)
    w = 300
    if len(diff) > w:
        s0 = int(np.argmax(np.convolve(diff, np.ones(w), "valid")))
        build_map(out["trip"], ref_arcs=ref,
                  pred_layers=[("Classifier-only route", out["clf_arcs"]), ("Predicted route (classifier + HMM)", out["arcs"])],
                  title=f"Student-test trip {tid} (detail)", focus=(s0, s0 + w)).save(str(MAPS / f"route_student_test_{tid}_detail.html"))
    print(tid, "saved")
