"""Generate candidates, reference labels and features for every train / student_test trip.
Output: data/processed/<partition>.pkl (one row per (observation, candidate arc))."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from multiprocessing import Pool
from pathsnap.data_io import ROOT, trip_ids, load_trip
from pathsnap.core import Trip, FEATURES

OUT = ROOT / "data" / "processed"; OUT.mkdir(parents=True, exist_ok=True)


def build(args):
    part, tid = args
    t0 = time.time()
    trip = Trip(load_trip(part, tid), tid)
    lab, pos, adist = trip.align_route()
    cand = trip.candidates()
    df = trip.features(cand)
    df["y"] = (df.arc.values == lab[df.obs.values]).astype(np.int8)
    df["trip"] = tid
    df["align_dist"] = adist[df.obs.values].astype(np.float32)
    stats = {"partition": part, "trip": tid, "n_obs": len(trip.gx), "n_rows": len(df),
             "cand_per_obs": len(df) / len(trip.gx),
             "miss_rate": 1 - df.groupby("obs").y.max().mean(),
             "align_dist_median": float(np.median(adist)), "align_dist_p95": float(np.percentile(adist, 95)),
             "align_dist_gt50": float((adist > 50).mean()),
             "miss_rate_reliable": 1 - df[df.align_dist <= 50].groupby("obs").y.max().mean() if (adist <= 50).any() else np.nan,
             "secs": time.time() - t0}
    keep = ["trip", "obs", "arc", "t", "y", "align_dist"] + FEATURES
    return df[keep], stats


if __name__ == "__main__":
    allstats = []
    for part in ["train", "student_test"]:
        jobs = [(part, t) for t in trip_ids(part)]
        with Pool(8) as p:
            res = p.map(build, jobs)
        df = pd.concat([r[0] for r in res], ignore_index=True)
        for c in FEATURES:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        df.to_pickle(OUT / f"{part}.pkl")
        allstats += [r[1] for r in res]
        print(part, df.shape, "positives per obs", df.groupby(["trip", "obs"]).y.sum().mean())
    st = pd.DataFrame(allstats)
    st.to_csv(ROOT / "results" / "candidate_stats.csv", index=False)
    print(st.groupby("partition")[["n_obs", "n_rows", "cand_per_obs", "miss_rate", "align_dist_median", "align_dist_p95", "align_dist_gt50"]].agg(["mean", "max"]).round(4).T)
    for part in ["train", "student_test"]:
        s = st[st.partition == part]
        print(part, "overall miss rate", round(float((s.miss_rate * s.n_obs).sum() / s.n_obs.sum()), 5))
        w = s.n_obs * (1 - s.align_dist_gt50)
        print(part, "miss rate on reliable labels", round(float((s.miss_rate_reliable * w).sum() / w.sum()), 5))
