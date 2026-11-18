"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model
(HistGradientBoosting) with 5-fold GroupKFold over the 80 training trips.
Objective: mean held-out macro F1 (candidate level, top-1 decision).
Then: refit tuned model, compare untuned vs tuned with the same protocol as 03_train_models.py."""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, joblib, optuna
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap.data_io import ROOT
from pathsnap.core import FEATURES
from pathsnap.evaluation import observation_metrics, top1_mask, RouteEvaluator, obs_keys

SEED = 42
N_TRIALS = 20
SEARCH_STRIDE = 4   # every 4th observation used for fitting inside the search (cost control)
RES = ROOT / "results"; FIG = RES / "figures"; ART = ROOT / "artifacts"; MOD = ART / "models"

BASE = dict(learning_rate=0.1, max_iter=300, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1,
            scoring="loss", n_iter_no_change=20, random_state=SEED)


def rows_for(df, stride):
    return np.flatnonzero((df.align_dist.values <= 50) & (df.obs.values % stride == 0))


def cv_score(tr, folds, params, stride):
    f1s, top1s, lls = [], [], []
    for tri, vai in folds:
        sub = tr.iloc[tri]; idx = rows_for(sub, stride)
        m = HistGradientBoostingClassifier(**params).fit(sub[FEATURES].values[idx], sub.y.values[idx])
        va = tr.iloc[vai]; va = va[(va.align_dist.values <= 50) & (va.obs.values % 2 == 1)]
        mm = observation_metrics(va, m.predict_proba(va[FEATURES].values)[:, 1])
        f1s.append(mm["macro_f1"]); top1s.append(mm["top1"]); lls.append(mm["log_loss"])
    return np.mean(f1s), np.std(f1s), np.mean(top1s), np.mean(lls)


if __name__ == "__main__":
    tr = pd.read_pickle(ROOT / "data/processed/train.pkl").sort_values(["trip", "obs", "arc"], kind="stable").reset_index(drop=True)
    st = pd.read_pickle(ROOT / "data/processed/student_test.pkl").sort_values(["trip", "obs", "arc"], kind="stable").reset_index(drop=True)
    tr["_key"] = obs_keys(tr); st["_key"] = obs_keys(st)
    fold_trips = json.load(open(RES / "cv_folds.json"))
    folds = []
    for k in range(5):
        va_mask = tr.trip.isin(fold_trips[f"fold{k}"]).values
        folds.append((np.flatnonzero(~va_mask), np.flatnonzero(va_mask)))

    def objective(trial):
        p = dict(BASE)
        p.update(
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 127, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 300, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
            max_features=trial.suggest_float("max_features", 0.4, 1.0),
            max_depth=trial.suggest_categorical("max_depth", [None, 6, 10, 16]),
            max_iter=500,
        )
        t0 = time.time()
        f1, f1s, top1, ll = cv_score(tr, folds, p, SEARCH_STRIDE)
        trial.set_user_attr("cv_top1", top1); trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("cv_f1_std", f1s); trial.set_user_attr("secs", time.time() - t0)
        print(f"trial {trial.number}: f1={f1:.4f} top1={top1:.4f} ({time.time()-t0:.0f}s) {trial.params}", flush=True)
        return f1

    storage = f"sqlite:///{ART / 'optuna_hgb.db'}"
    study = optuna.create_study(study_name="hgb_tpe", direction="maximize", storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8))
    # trial 0 = untuned configuration evaluated under the same search protocol
    if len(study.trials) == 0:
        study.enqueue_trial({"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                             "l2_regularization": 1e-4, "max_features": 1.0, "max_depth": None})
    t_search = time.time()
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    study.optimize(objective, n_trials=max(0, N_TRIALS - n_done))
    t_search = time.time() - t_search
    trials = study.trials_dataframe()
    trials = trials[trials.state == "COMPLETE"].reset_index(drop=True)
    trials.to_csv(RES / "bayes_opt_trials.csv", index=False)
    best = study.best_params
    print("best", best, study.best_value)

    # optimisation history plot
    vals = trials.value.values
    plt.figure(figsize=(7, 4))
    plt.plot(trials.number, vals, "o", alpha=.6, label="trial CV macro F1")
    plt.plot(trials.number, np.maximum.accumulate(vals), "-", color="tab:red", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV macro F1 (5-fold GroupKFold)"); plt.title("Bayesian optimisation (TPE) history")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "bayes_opt_history.png", dpi=130); plt.close()
    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(6, 3.5)); plt.barh(list(imp.keys())[::-1], list(imp.values())[::-1])
        plt.xlabel("importance (fANOVA)"); plt.title("Hyperparameter importance"); plt.tight_layout()
        plt.savefig(FIG / "bayes_opt_importance.png", dpi=130); plt.close()
        json.dump(imp, open(RES / "bayes_opt_importance.json", "w"), indent=1)
    except Exception as e:
        print("importance failed", e)

    # untuned vs tuned under the full protocol (same as model comparison: stride 2)
    tuned = dict(BASE); tuned.update(best); tuned["max_iter"] = 500
    route_eval = RouteEvaluator("student_test", sorted(st.trip.unique()))
    out = []
    for name, params in [("HGB untuned", BASE), ("HGB tuned", tuned)]:
        f1, f1s, top1, ll = cv_score(tr, folds, params, 2)
        idx = rows_for(tr, 2)
        t0 = time.time(); m = HistGradientBoostingClassifier(**params).fit(tr[FEATURES].values[idx], tr.y.values[idx]); tfit = time.time() - t0
        t0 = time.time(); p = m.predict_proba(st[FEATURES].values)[:, 1]; tinf = time.time() - t0
        mm = observation_metrics(st, p); rs = route_eval.scores(st, top1_mask(st, p))
        out.append({"model": name, "cv_macro_f1": f1, "cv_macro_f1_std": f1s, "cv_top1": top1, "cv_log_loss": ll,
                    "st_top1": mm["top1"], "st_macro_f1": mm["macro_f1"], "st_log_loss": mm["log_loss"],
                    "st_continuity": rs.continuity.mean(), "train_time_s": tfit,
                    "infer_ms_per_1k_obs": tinf / mm["n_obs"] * 1e6, "n_iter": int(m.n_iter_)})
        print(out[-1], flush=True)
        if name == "HGB tuned":
            joblib.dump(m, MOD / "HGB_tuned.joblib", compress=3)
            np.save(ROOT / "data/processed/student_test_probs_tuned.npy", p)
            json.dump({"train_score": list(map(float, m.train_score_)), "validation_score": list(map(float, m.validation_score_))},
                      open(RES / "history_HGB_tuned.json", "w"))
    cmp = pd.DataFrame(out); cmp.to_csv(RES / "tuned_vs_untuned.csv", index=False)
    json.dump({"best_params": best, "best_cv_macro_f1_search": study.best_value, "n_trials": int(len(trials)),
               "search_seconds": t_search, "search_fit_stride": SEARCH_STRIDE,
               "tuned_params_full": {k: v for k, v in tuned.items()}}, open(RES / "bayes_opt_summary.json", "w"), indent=1, default=str)
    print(cmp.round(4).to_string())
