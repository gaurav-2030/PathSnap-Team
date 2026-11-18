"""PathSnap FastAPI backend (Laptop 1).

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Evaluation trips are read from data/raw/evaluator_test/<trip_id>/ (override with PATHSNAP_TRIPS_DIR).
The reference route is read only inside /predict to score the prediction; it is never passed
to the predictor and never returned to the client."""
import os, sys, time, logging
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.core import Trip, point_seg, route_set_scores
from pathsnap.pipeline import PathSnapPredictor, route_geometry

TRIPS_DIR = Path(os.environ.get("PATHSNAP_TRIPS_DIR", ROOT / "data" / "raw" / "evaluator_test"))
LOG_DIR = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / "api_requests.log")):
    h.setFormatter(_fmt); log.addHandler(h)

app = FastAPI(title="PathSnap Map-Matching API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
predictor = PathSnapPredictor()
log.info("model loaded: %s | HMM %s | trips dir: %s", predictor.cfg["model_name"], predictor.hmm, TRIPS_DIR)


def _read(path, names):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")


def list_trips():
    if not TRIPS_DIR.exists():
        return []
    return sorted(p.name for p in TRIPS_DIR.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())


def load_inputs(tid):
    d = TRIPS_DIR / tid
    return {"track": _read(d / f"{tid}.track", ["lon", "lat", "t"]),
            "nodes": _read(d / f"{tid}.nodes", ["lon", "lat"]),
            "arcs": _read(d / f"{tid}.arcs", ["u", "v"]),
            "route": None}


def score_against_reference(tid, trip, out):
    """Server-side scoring with the reference route (kept inside the API)."""
    f = TRIPS_DIR / tid / f"{tid}.route"
    if not f.exists():
        return None
    ref = Trip({"track": trip.track, "nodes": pd.DataFrame({"lon": trip.node_lon, "lat": trip.node_lat}),
                "arcs": pd.DataFrame({"u": trip.u, "v": trip.v}),
                "route": _read(f, ["arc_id"])}, tid)
    lab, _, _ = ref.align_route()
    lab = lab[out["obs"]]
    hmm_acc = float(np.mean(out["arcs"] == lab))
    clf_acc = float(np.mean(out["clf_arcs"] == lab))
    seq = out["arcs"][np.r_[True, out["arcs"][1:] != out["arcs"][:-1]]]
    p, r, f1 = route_set_scores(seq, ref.route)
    return {"top1_accuracy_hmm": hmm_acc, "top1_accuracy_classifier_only": clf_acc,
            "route_precision": p, "route_recall": r, "route_f1": f1}


class PredictRequest(BaseModel):
    trip_id: str


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.time()
    resp = await call_next(request)
    log.info("%s %s from %s -> %d (%.0f ms)", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code, 1000 * (time.time() - t0))
    return resp


@app.get("/health")
def health():
    return {"status": "ok", "model": predictor.cfg["model_name"], "n_trips": len(list_trips())}


@app.get("/trips")
def trips():
    return {"trips": list_trips()}


@app.post("/predict")
def predict(req: PredictRequest):
    tid = req.trip_id.strip()
    if tid not in list_trips():
        raise HTTPException(404, f"unknown trip {tid}")
    log.info("PREDICT trip=%s received", tid)
    t0 = time.time()
    out = predictor.predict(load_inputs(tid), tid)
    trip = out["trip"]
    metrics = score_against_reference(tid, trip, out)
    segs, seq = route_geometry(trip, out["arcs"])
    step = max(1, len(trip.track) // 1500)
    gps = trip.track[["lat", "lon"]].values[::step].round(6).tolist()
    elapsed = time.time() - t0
    log.info("PREDICT trip=%s obs=%d candidates=%d route_arcs=%d acc_hmm=%s acc_clf=%s continuity=%.4f time=%.2fs",
             tid, len(out["obs"]), out["n_candidates"], len(seq),
             f"{metrics['top1_accuracy_hmm']:.4f}" if metrics else "n/a",
             f"{metrics['top1_accuracy_classifier_only']:.4f}" if metrics else "n/a",
             out["continuity"], elapsed)
    return {
        "trip_id": tid,
        "n_observations": int(len(out["obs"])),
        "n_candidates": out["n_candidates"],
        "predicted_arc_ids": [int(a) for a in out["arcs"]],
        "predicted_route_arcs": [int(a) for a in seq],
        "predicted_route_segments": segs,
        "gps_points": gps,
        "metrics": metrics,
        "route_continuity": out["continuity"],
        "chain_breaks": out["breaks"],
        "timing_s": {k: round(v, 3) for k, v in out["timing"].items()} | {"total_request": round(elapsed, 3)},
        "model": predictor.cfg["model_name"],
    }
