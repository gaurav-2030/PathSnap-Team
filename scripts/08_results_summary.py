"""Print the model comparison, tuning and HMM tables from results/ (no retraining)."""
from pathlib import Path
import json
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def show(title, df):
    print(f"\n=== {title} ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


models = pd.read_csv(RES / "model_comparison.csv")
show("Seven models (5-fold GroupKFold CV + student-test)",
     models[["model", "family", "cv_macro_f1", "cv_macro_f1_std", "st_top1", "st_macro_f1", "st_log_loss",
             "train_time_s", "infer_ms_per_1k_obs"]].sort_values("st_top1", ascending=False))

show("Untuned vs Bayesian-tuned HistGradientBoosting",
     pd.read_csv(RES / "tuned_vs_untuned.csv")[["model", "cv_macro_f1", "st_top1", "st_macro_f1", "st_log_loss"]])

best = json.loads((RES / "bayes_opt_summary.json").read_text())
print("\n=== Best hyper-parameters (Optuna TPE, %d trials) ===" % best["n_trials"])
for k, v in best["best_params"].items():
    print(f"  {k:20s} {v:.4g}" if isinstance(v, float) else f"  {k:20s} {v}")

show("Decoder comparison on student-test",
     pd.read_csv(RES / "hmm_comparison_student_test.csv")[["variant", "st_top1", "st_macro_f1", "continuity", "route_f1"]])
