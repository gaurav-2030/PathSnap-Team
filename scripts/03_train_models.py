"""Train and compare candidate classifiers.
- 5-fold GroupKFold over the 80 training trips (groups = trip id)
- every model refit on all training trips and evaluated on the 10 student-test trips
Outputs go to results/ and artifacts/models/."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from pathsnap.data_io import ROOT
from pathsnap.core import FEATURES
from pathsnap.evaluation import observation_metrics, confusion, top1_mask, RouteEvaluator, obs_keys

SEED = 42
RES = ROOT / "results"; FIG = RES / "figures"; MOD = ROOT / "artifacts" / "models"
for d in (FIG, MOD): d.mkdir(parents=True, exist_ok=True)


def models():
    return {
        "LogisticRegression": dict(family="Linear", fit_obs_stride=2, max_rows=None,
            make=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            config="StandardScaler + L2, C=1.0, lbfgs"),
        "GaussianNB": dict(family="Probabilistic", fit_obs_stride=2, max_rows=None,
            make=lambda: GaussianNB(var_smoothing=1e-9), config="var_smoothing=1e-9"),
        "KNN": dict(family="Instance-based", fit_obs_stride=2, max_rows=150_000,
            make=lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
            config="k=25, distance-weighted, Euclidean on standardised features, 150k training rows"),
        "DecisionTree": dict(family="Tree-based", fit_obs_stride=2, max_rows=None,
            make=lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=SEED),
            config="max_depth=12, min_samples_leaf=20"),
        "RandomForest": dict(family="Ensemble (bagging)", fit_obs_stride=2, max_rows=None,
            make=lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=5, max_features="sqrt",
                                                max_samples=0.3, n_jobs=-1, random_state=SEED),
            config="150 trees, min_samples_leaf=5, max_features=sqrt, max_samples=0.3"),
        "HistGradientBoosting": dict(family="Boosting", fit_obs_stride=2, max_rows=None,
            make=lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                                        early_stopping=True, validation_fraction=0.1,
                                                        scoring="loss", n_iter_no_change=20, random_state=SEED),
            config="lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping (10% val)"),
        "MLP": dict(family="Neural network", fit_obs_stride=2, max_rows=400_000,
            make=lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                       batch_size=512, learning_rate_init=1e-3, max_iter=60, early_stopping=True,
                                       validation_fraction=0.1, n_iter_no_change=8, random_state=SEED)),
            config="hidden=(64,32), ReLU, Adam lr=1e-3, alpha=1e-4, early stopping, 400k rows"),
    }


def fit_rows(df, stride, max_rows, rng):
    """Training rows: reliable reference labels only, every `stride`-th observation."""
    m = (df.align_dist.values <= 50) & (df.obs.values % stride == 0)
    idx = np.flatnonzero(m)
    if max_rows and len(idx) > max_rows:
        # sample whole observations so every kept observation retains its full candidate set
        keys = df["_key"].values[idx]
        uk = np.unique(keys)
        n_keep = int(len(uk) * max_rows / len(idx))
        keep = set(rng.choice(uk, n_keep, replace=False))
        idx = idx[np.fromiter((k in keep for k in keys), bool, len(keys))]
    return idx


def proba(model, X):
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    tr = pd.read_pickle(ROOT / "data/processed/train.pkl").sort_values(["trip", "obs", "arc"], kind="stable").reset_index(drop=True)
    st = pd.read_pickle(ROOT / "data/processed/student_test.pkl").sort_values(["trip", "obs", "arc"], kind="stable").reset_index(drop=True)
    tr["_key"] = obs_keys(tr); st["_key"] = obs_keys(st)
    print("train rows", len(tr), "student rows", len(st))
    groups = tr.trip.values
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(tr, groups=groups))
    json.dump({f"fold{k}": sorted(set(groups[va])) for k, (_, va) in enumerate(folds)}, open(RES / "cv_folds.json", "w"), indent=1)
    route_eval = RouteEvaluator("student_test", sorted(st.trip.unique()))
    ref_cont = route_eval.reference_continuity(st)
    ref_cont.to_csv(RES / "reference_continuity_student_test.csv", index=False)
    print("reference-label continuity (student test):", ref_cont.continuity.mean().round(4))

    rows, fold_rows, conf = [], [], {}
    st_probs = {}
    Xst = st[FEATURES].values
    # nearest-candidate heuristic baseline (no learning) for reference
    base = -st.dist.values
    m_top = observation_metrics(st, base)
    rs = route_eval.scores(st, top1_mask(st, base))
    st_probs["NearestCandidate"] = base
    rows.append({"model": "Nearest-candidate baseline", "family": "Heuristic (no training)", "config": "argmin perpendicular distance",
                 "cv_macro_f1": np.nan, "cv_macro_f1_std": np.nan, "cv_top1": np.nan,
                 "st_top1": m_top["top1"], "st_macro_f1": m_top["macro_f1"], "st_macro_precision": m_top["macro_precision"],
                 "st_macro_recall": m_top["macro_recall"], "st_log_loss": np.nan,
                 "st_continuity": rs.continuity.mean(), "st_route_f1": rs.route_f1.mean(),
                 "train_time_s": 0.0, "infer_ms_per_1k_obs": np.nan})

    rng = np.random.default_rng(SEED)
    for name, spec in models().items():
        print("==", name, flush=True)
        cv = []
        for k, (tri, vai) in enumerate(folds):
            sub = tr.iloc[tri]
            idx = fit_rows(sub, spec["fit_obs_stride"], spec["max_rows"], rng)
            mdl = spec["make"]()
            t0 = time.time(); mdl.fit(sub[FEATURES].values[idx], sub.y.values[idx]); tfit = time.time() - t0
            va = tr.iloc[vai]
            va = va[va.align_dist.values <= 50]
            va = va[va.obs.values % 2 == 1]          # held-out evaluation subset (alternate observations)
            p = proba(mdl, va[FEATURES].values)
            mm = observation_metrics(va, p)
            mm.update(model=name, fold=k, fit_s=tfit, fit_rows=len(idx)); cv.append(mm); fold_rows.append(mm)
            print(f"  fold {k}: top1={mm['top1']:.4f} f1={mm['macro_f1']:.4f} ll={mm['log_loss']:.4f} ({tfit:.1f}s)", flush=True)
        cvdf = pd.DataFrame(cv)
        # refit on all 80 training trips
        idx = fit_rows(tr, spec["fit_obs_stride"], spec["max_rows"], rng)
        mdl = spec["make"]()
        t0 = time.time(); mdl.fit(tr[FEATURES].values[idx], tr.y.values[idx]); tfit = time.time() - t0
        t0 = time.time(); p = proba(mdl, Xst); tinf = time.time() - t0
        mm = observation_metrics(st, p)
        pick = top1_mask(st, p)
        rs = route_eval.scores(st, pick)
        tr_idx = fit_rows(tr, 4, None, rng)  # training-fit check on a quarter of training observations
        mtr = observation_metrics(tr.iloc[tr_idx], proba(mdl, tr[FEATURES].values[tr_idx]))
        conf[name] = confusion(st, p).tolist()
        st_probs[name] = p
        joblib.dump(mdl, MOD / f"{name}.joblib", compress=3)
        rows.append({"model": name, "family": spec["family"], "config": spec["config"],
                     "cv_macro_f1": cvdf.macro_f1.mean(), "cv_macro_f1_std": cvdf.macro_f1.std(), "cv_top1": cvdf.top1.mean(),
                     "cv_log_loss": cvdf.log_loss.mean(),
                     "train_top1": mtr["top1"], "train_macro_f1": mtr["macro_f1"], "train_log_loss": mtr["log_loss"],
                     "st_top1": mm["top1"], "st_macro_f1": mm["macro_f1"], "st_macro_precision": mm["macro_precision"],
                     "st_macro_recall": mm["macro_recall"], "st_log_loss": mm["log_loss"],
                     "st_continuity": rs.continuity.mean(), "st_route_f1": rs.route_f1.mean(),
                     "train_time_s": tfit, "infer_ms_per_1k_obs": tinf / mm["n_obs"] * 1e6,
                     "fit_rows": len(idx)})
        print(f"  student-test: top1={mm['top1']:.4f} f1={mm['macro_f1']:.4f} ll={mm['log_loss']:.4f} cont={rs.continuity.mean():.4f}", flush=True)
        # epoch-wise histories (only models that really produce them)
        est = mdl[-1] if hasattr(mdl, "steps") else mdl
        if name == "MLP":
            json.dump({"loss_curve": list(map(float, est.loss_curve_)), "validation_scores": list(map(float, est.validation_scores_))},
                      open(RES / "history_MLP.json", "w"))
        if name == "HistGradientBoosting":
            json.dump({"train_score": list(map(float, est.train_score_)), "validation_score": list(map(float, est.validation_score_))},
                      open(RES / "history_HistGradientBoosting.json", "w"))

    comp = pd.DataFrame(rows)
    comp.to_csv(RES / "model_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RES / "cv_fold_scores.csv", index=False)
    json.dump(conf, open(RES / "confusion_matrices.json", "w"), indent=1)
    np.savez_compressed(ROOT / "data/processed/student_test_probs.npz", **st_probs)
    pd.set_option("display.width", 250)
    print(comp.drop(columns=["config"]).round(4).to_string())
