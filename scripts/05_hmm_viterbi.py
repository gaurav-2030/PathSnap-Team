"""HMM / Viterbi route reconstruction on top of the tuned classifier.
1. Out-of-fold classifier probabilities for the 80 training trips (same 5 GroupKFold folds).
2. Decoder parameters (beta, emission weight, gap threshold) tuned on training trips (OOF probabilities).
3. Student-test comparison: classifier-only vs classifier+HMM (and a distance-only HMM reference).
4. Export final artifacts/final_config.json used by the API."""
import sys, time, json, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap.data_io import ROOT, load_trip
from pathsnap.core import Trip, FEATURES, CAND_RADIUS, CAND_K, pair_table, log_transitions, viterbi, continuity, route_set_scores
from pathsnap.evaluation import metrics_from_pick, obs_keys, group_bounds

RES = ROOT / "results"; FIG = RES / "figures"; ART = ROOT / "artifacts"
SEED = 42
BETAS = [2, 4, 8, 16]
EMIS_W = [0.5, 1.0, 2.0]
GAPS = [30, 60]


def per_trip(df):
    for t, g in df.groupby("trip", sort=True):
        yield t, g


def decode_rows(g, logp, tab, beta, w, gap):
    logt = log_transitions(tab, beta, gap) if tab["n_obs"] > 1 else None
    rows, br = viterbi(logp, tab, logt, w)
    return rows, br


def argmax_rows(tab, score):
    return np.array([s + int(np.argmax(score[s:e])) for s, e in zip(tab["starts"], tab["ends"])])


if __name__ == "__main__":
    summary = json.load(open(RES / "bayes_opt_summary.json"))
    params = summary["tuned_params_full"]
    params = {k: (None if v in ("None", None) else v) for k, v in params.items()}
    tr = pd.read_pickle(ROOT / "data/processed/train.pkl").sort_values(["trip", "obs", "arc"], kind="stable").reset_index(drop=True)
    st = pd.read_pickle(ROOT / "data/processed/student_test.pkl").sort_values(["trip", "obs", "arc"], kind="stable").reset_index(drop=True)
    tr["_key"] = obs_keys(tr); st["_key"] = obs_keys(st)

    # 1. out-of-fold probabilities
    oof_path = ROOT / "data/processed/train_oof_probs_tuned.npy"
    if oof_path.exists():
        oof = np.load(oof_path)
    else:
        fold_trips = json.load(open(RES / "cv_folds.json"))
        oof = np.zeros(len(tr))
        for k in range(5):
            va = tr.trip.isin(fold_trips[f"fold{k}"]).values
            sub = tr[~va]; idx = np.flatnonzero((sub.align_dist.values <= 50) & (sub.obs.values % 2 == 0))
            m = HistGradientBoostingClassifier(**params).fit(sub[FEATURES].values[idx], sub.y.values[idx])
            oof[va] = m.predict_proba(tr.loc[va, FEATURES].values)[:, 1]
            print("oof fold", k, flush=True)
        np.save(oof_path, oof)
    st_prob = np.load(ROOT / "data/processed/student_test_probs_tuned.npy")

    # 2. decoder tuning on training trips (reliable labels only for scoring)
    sigma = 1.4826 * float(np.median(tr.loc[(tr.y == 1) & (tr.align_dist <= 50), "dist"]))
    trips_tr = sorted(tr.trip.unique())[::2]   # 40 of the 80 training trips for the decoder grid
    cache = {}
    t0 = time.time()
    for t, g in per_trip(tr):
        if t not in trips_tr:
            continue
        trip = Trip(load_trip("train", t), t)
        cache[t] = (g.index.values, pair_table(trip, g))
    print(f"train pair tables built in {time.time()-t0:.0f}s", flush=True)
    grid = []
    for beta, w, gap in itertools.product(BETAS, EMIS_W, GAPS):
        correct = 0; n = 0
        for t in trips_tr:
            idx, tab = cache[t]
            g = tr.loc[idx]
            rows, _ = decode_rows(g, np.log(np.clip(oof[idx], 1e-6, 1)), tab, beta, w, gap)
            rel = g.align_dist.values[rows] <= 50
            correct += int((g.y.values[rows] == 1)[rel].sum()); n += int(rel.sum())
        grid.append({"beta": beta, "emis_w": w, "gap_s": gap, "train_oof_top1": correct / n})
        print(grid[-1], flush=True)
    grid = pd.DataFrame(grid); grid.to_csv(RES / "hmm_decoder_grid.csv", index=False)
    bestg = grid.sort_values("train_oof_top1", ascending=False).iloc[0]
    beta, w, gap = float(bestg.beta), float(bestg.emis_w), float(bestg.gap_s)
    # classifier-only OOF reference on the same observations
    c_ok = 0; c_n = 0
    for t in trips_tr:
        idx, tab = cache[t]; g = tr.loc[idx]
        rows = argmax_rows(tab, oof[idx]); rel = g.align_dist.values[rows] <= 50
        c_ok += int((g.y.values[rows] == 1)[rel].sum()); c_n += int(rel.sum())
    print("train OOF classifier-only top1", c_ok / c_n, "best decoder", dict(bestg))

    # 3. student-test comparison
    variants = {
        "Nearest candidate (no HMM)": ("argmax", lambda g, p: -g.dist.values),
        "Distance-only HMM (Gaussian emission)": ("hmm", lambda g, p: -0.5 * (g.dist.values / sigma) ** 2),
        "Classifier only (tuned HGB)": ("argmax", lambda g, p: p),
        "Classifier + HMM/Viterbi": ("hmm", lambda g, p: np.log(np.clip(p, 1e-6, 1))),
    }
    st_trips = {}
    for t, g in per_trip(st):
        trip = Trip(load_trip("student_test", t), t)
        st_trips[t] = (trip, g.index.values, pair_table(trip, g))
    res, per = [], []
    picks = {}
    for vname, (kind, score_fn) in variants.items():
        pick = np.zeros(len(st), dtype=np.int8)
        tdec = 0.0; brk = 0
        for t, (trip, idx, tab) in st_trips.items():
            g = st.loc[idx]; sc = score_fn(g, st_prob[idx])
            t0 = time.time()
            if kind == "hmm":
                rows, b = decode_rows(g, sc, tab, beta, 1.0 if vname.startswith("Distance") else w, gap); brk += b
            else:
                rows = argmax_rows(tab, sc)
            tdec += time.time() - t0
            pick[idx[rows]] = 1
            obs, arcs = g.obs.values[rows], g.arc.values[rows]
            seq = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
            pr, rc, f1 = route_set_scores(seq, trip.route)
            mt = metrics_from_pick(g, pick[idx])
            per.append({"variant": vname, "trip": t, "top1": mt["top1"], "macro_f1": mt["macro_f1"],
                        "continuity": continuity(trip, arcs, obs), "route_precision": pr, "route_recall": rc, "route_f1": f1})
        m = metrics_from_pick(st, pick)
        pt = pd.DataFrame([p for p in per if p["variant"] == vname])
        res.append({"variant": vname, "st_top1": m["top1"], "st_macro_f1": m["macro_f1"],
                    "st_macro_precision": m["macro_precision"], "st_macro_recall": m["macro_recall"],
                    "continuity": pt.continuity.mean(), "route_precision": pt.route_precision.mean(),
                    "route_recall": pt.route_recall.mean(), "route_f1": pt.route_f1.mean(),
                    "chain_breaks": brk, "decode_s_total": tdec})
        picks[vname] = pick
        print(res[-1], flush=True)
    res = pd.DataFrame(res); res.to_csv(RES / "hmm_comparison_student_test.csv", index=False)
    per = pd.DataFrame(per); per.to_csv(RES / "hmm_per_trip_student_test.csv", index=False)
    np.savez_compressed(ROOT / "data/processed/student_test_picks.npz", **{k.split(" ")[0] + str(i): v for i, (k, v) in enumerate(picks.items())})

    # plot classifier vs HMM per trip
    a = per[per.variant == "Classifier only (tuned HGB)"].set_index("trip")
    b = per[per.variant == "Classifier + HMM/Viterbi"].set_index("trip")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(a)); wd = 0.38
    ax[0].bar(x - wd/2, a.top1, wd, label="classifier only"); ax[0].bar(x + wd/2, b.top1, wd, label="classifier + HMM")
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(int(t)) for t in a.index], rotation=45); ax[0].set_ylim(0.5, 1.0)
    ax[0].set_title("Student-test Top-1 accuracy per trip"); ax[0].set_xlabel("trip"); ax[0].legend()
    ax[1].bar(x - wd/2, a.continuity, wd, label="classifier only"); ax[1].bar(x + wd/2, b.continuity, wd, label="classifier + HMM")
    ax[1].set_xticks(x); ax[1].set_xticklabels([str(int(t)) for t in a.index], rotation=45); ax[1].set_ylim(0.5, 1.01)
    ax[1].set_title("Route continuity per trip"); ax[1].set_xlabel("trip"); ax[1].legend()
    plt.tight_layout(); plt.savefig(FIG / "hmm_vs_classifier_per_trip.png", dpi=130); plt.close()

    # 4. final config for deployment
    cfg = {"model_file": "HGB_tuned.joblib", "model_name": "HistGradientBoosting (Bayesian-tuned)",
           "features": FEATURES, "cand_radius": CAND_RADIUS, "cand_k": CAND_K,
           "hmm": {"beta": beta, "emis_w": w, "gap_s": gap}, "gaussian_sigma_m": sigma,
           "classifier_params": params}
    json.dump(cfg, open(ART / "final_config.json", "w"), indent=1, default=str)
    print(res.round(4).to_string())
