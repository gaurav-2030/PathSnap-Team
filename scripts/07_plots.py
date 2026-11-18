"""Comparison plots from the recorded experiment results (results/*.csv / *.json)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathsnap.data_io import ROOT

RES = ROOT / "results"; FIG = RES / "figures"
comp = pd.read_csv(RES / "model_comparison.csv")
ml = comp[comp.family != "Heuristic (no training)"].reset_index(drop=True)
folds = pd.read_csv(RES / "cv_fold_scores.csv")
names = ml.model.tolist(); x = np.arange(len(names))

# 1. CV macro F1 with per-fold points
plt.figure(figsize=(9, 4))
plt.bar(x, ml.cv_macro_f1, yerr=ml.cv_macro_f1_std, capsize=4, color="#4c78a8")
for i, n in enumerate(names):
    f = folds[folds.model == n].macro_f1.values
    plt.scatter(np.full(len(f), i) + np.linspace(-.15, .15, len(f)), f, color="k", s=10, zorder=3)
plt.xticks(x, names, rotation=20); plt.ylim(0.88, 0.97); plt.ylabel("macro F1")
plt.title("5-fold GroupKFold CV macro F1 (bars = mean ± sd, dots = folds)"); plt.tight_layout()
plt.savefig(FIG / "cv_macro_f1.png", dpi=130); plt.close()

# 2. student-test top-1 and macro F1
w = 0.38
plt.figure(figsize=(9, 4))
plt.bar(x - w/2, ml.st_top1, w, label="Top-1 accuracy"); plt.bar(x + w/2, ml.st_macro_f1, w, label="Macro F1")
base = comp[comp.family == "Heuristic (no training)"]
if len(base):
    plt.axhline(base.st_top1.iloc[0], ls="--", color="gray", label="nearest-candidate Top-1")
for i in range(len(names)):
    plt.text(i - w/2, ml.st_top1[i] + .003, f"{ml.st_top1[i]:.3f}", ha="center", fontsize=7)
    plt.text(i + w/2, ml.st_macro_f1[i] + .003, f"{ml.st_macro_f1[i]:.3f}", ha="center", fontsize=7)
plt.xticks(x, names, rotation=20); plt.ylim(0.75, 1.0); plt.legend(loc="lower right", fontsize=8)
plt.title("Student-test (10 trips) observation-level performance"); plt.tight_layout()
plt.savefig(FIG / "student_test_top1_f1.png", dpi=130); plt.close()

# 3. log loss (CV and student test)
plt.figure(figsize=(9, 4))
plt.bar(x - w/2, ml.cv_log_loss, w, label="CV log loss"); plt.bar(x + w/2, ml.st_log_loss, w, label="student-test log loss")
plt.yscale("log"); plt.xticks(x, names, rotation=20); plt.ylabel("binary log loss (log scale)"); plt.legend()
plt.title("Probability quality (lower is better)"); plt.tight_layout(); plt.savefig(FIG / "log_loss.png", dpi=130); plt.close()

# 4. train vs CV vs student (over/under-fitting check) and efficiency
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(names, ml.train_macro_f1, "o-", label="training fit"); ax[0].plot(names, ml.cv_macro_f1, "s-", label="CV (held-out trips)")
ax[0].plot(names, ml.st_macro_f1, "^-", label="student-test"); ax[0].set_ylabel("macro F1"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=20)
ax[0].set_title("Generalisation gap")
ax[1].scatter(ml.infer_ms_per_1k_obs, ml.st_top1, s=60)
for i, n in enumerate(names): ax[1].annotate(n, (ml.infer_ms_per_1k_obs[i], ml.st_top1[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax[1].set_xscale("log"); ax[1].set_xlabel("inference time (ms per 1000 observations, log)"); ax[1].set_ylabel("student-test Top-1")
ax[1].set_title("Accuracy vs inference cost")
plt.tight_layout(); plt.savefig(FIG / "generalisation_and_speed.png", dpi=130); plt.close()

# 5. route continuity
plt.figure(figsize=(9, 3.6))
plt.bar(np.arange(len(comp)), comp.st_continuity, color="#72b7b2")
plt.xticks(np.arange(len(comp)), comp.model, rotation=20); plt.ylim(0.9, 1.0); plt.ylabel("continuity")
plt.title("Student-test route continuity of top-1 predictions"); plt.tight_layout()
plt.savefig(FIG / "route_continuity_models.png", dpi=130); plt.close()

# 6. confusion matrices
conf = json.load(open(RES / "confusion_matrices.json"))
n = len(conf); fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 2.9))
for a, (k, cm) in zip(ax, conf.items()):
    cm = np.array(cm); a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=7, color="w" if cm[i, j] > cm.max() / 2 else "k")
    a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["pred 0", "pred 1"], fontsize=7); a.set_yticklabels(["true 0", "true 1"], fontsize=7)
    a.set_title(k, fontsize=8)
plt.suptitle("Candidate-level confusion matrices, student-test (top-1 decision)", fontsize=9); plt.tight_layout()
plt.savefig(FIG / "confusion_matrices.png", dpi=130); plt.close()

# 7. real training histories
if (RES / "history_MLP.json").exists():
    h = json.load(open(RES / "history_MLP.json"))
    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax1.plot(np.arange(1, len(h["loss_curve"]) + 1), h["loss_curve"], "o-", label="training loss (log loss)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss")
    ax2 = ax1.twinx(); ax2.plot(np.arange(1, len(h["validation_scores"]) + 1), h["validation_scores"], "s-", color="tab:orange", label="validation accuracy")
    ax2.set_ylabel("validation accuracy (10% hold-out)")
    fig.legend(loc="center right", fontsize=8); plt.title("MLP training history (final fit)"); plt.tight_layout()
    plt.savefig(FIG / "history_mlp.png", dpi=130); plt.close()
for tag in ["HistGradientBoosting", "HGB_tuned"]:
    p = RES / f"history_{tag}.json"
    if p.exists():
        h = json.load(open(p))
        plt.figure(figsize=(6, 3.6))
        plt.plot(-np.array(h["train_score"]), label="training log loss"); plt.plot(-np.array(h["validation_score"]), label="validation log loss")
        plt.xlabel("boosting iteration"); plt.ylabel("log loss"); plt.legend(); plt.title(f"{tag}: loss per boosting iteration")
        plt.tight_layout(); plt.savefig(FIG / f"history_{tag}.png", dpi=130); plt.close()
print("plots written")
