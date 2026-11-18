"""End-to-end inference used by the API: raw trip -> candidates -> features ->
classifier probabilities -> HMM/Viterbi -> matched road segments.
The predictor never receives the reference route."""
import json, time
from pathlib import Path
import numpy as np
import joblib

from .core import Trip, FEATURES, pair_table, log_transitions, viterbi, continuity

ART = Path(__file__).resolve().parents[1] / "artifacts"


class PathSnapPredictor:
    def __init__(self, art_dir=ART):
        art_dir = Path(art_dir)
        self.cfg = json.load(open(art_dir / "final_config.json"))
        self.model = joblib.load(art_dir / "models" / self.cfg["model_file"])
        self.hmm = self.cfg["hmm"]

    def predict(self, raw, tid=""):
        assert raw.get("route") is None, "reference route must not be passed to the predictor"
        t0 = time.time()
        trip = Trip(raw, tid)
        cand = trip.candidates(radius=self.cfg["cand_radius"], k=self.cfg["cand_k"])
        feats = trip.features(cand)
        prob = self.model.predict_proba(feats[FEATURES].values.astype(np.float32))[:, 1]
        t1 = time.time()
        tab = pair_table(trip, feats)
        logt = log_transitions(tab, self.hmm["beta"], self.hmm["gap_s"]) if tab["n_obs"] > 1 else None
        rows, breaks = viterbi(np.log(np.clip(prob, 1e-6, 1)), tab, logt, self.hmm["emis_w"])
        t2 = time.time()
        starts, ends = tab["starts"], tab["ends"]
        clf_rows = np.array([s + int(np.argmax(prob[s:e])) for s, e in zip(starts, ends)])
        obs = feats.obs.values[rows]
        arcs = feats.arc.values[rows]
        return {
            "trip": trip, "obs": obs, "arcs": arcs, "clf_arcs": feats.arc.values[clf_rows],
            "conf": prob[rows], "breaks": int(breaks),
            "continuity": continuity(trip, arcs, obs),
            "timing": {"classifier_s": t1 - t0, "viterbi_s": t2 - t1, "total_s": t2 - t0},
            "n_candidates": int(len(feats)),
        }


def route_geometry(trip, arcs):
    """Ordered list of segments [[lat,lon],[lat,lon]] of distinct consecutive arcs."""
    seq = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
    return [[[float(trip.node_lat[trip.u[a]]), float(trip.node_lon[trip.u[a]])],
             [float(trip.node_lat[trip.v[a]]), float(trip.node_lon[trip.v[a]])]] for a in seq], seq
