# Models and Bayesian optimisation

Scripts: `scripts/03_train_models.py` (7 models), `scripts/04_bayes_opt.py` (Optuna), `scripts/07_plots.py` (figures), `scripts/08_results_summary.py` (prints the tables below from `results/`).

## Setup
- Task: binary classification of (GPS observation, candidate arc) pairs, 22 features; the top-scoring candidate per observation is the prediction.
- Validation: 5-fold GroupKFold grouped by trip (fold membership in `results/cv_folds.json`), then refit on all 80 train trips and tested on the 10 student-test trips.
- Metrics: Top-1 accuracy per observation, macro F1 over correct/incorrect candidates using the top-1 decision, log loss, route continuity, route F1.

## Seven models (`results/model_comparison.csv`)
| Model | Family | CV macro F1 | Student-test Top-1 | Student-test log loss |
|---|---|---|---|---|
| Nearest candidate (no training) | heuristic | - | 0.543 | - |
| LogisticRegression | linear | 0.937 | 0.883 | 0.082 |
| GaussianNB | probabilistic | 0.925 | 0.858 | 0.712 |
| KNN (k=25) | instance-based | 0.931 | 0.876 | 0.091 |
| DecisionTree | tree | 0.935 | 0.890 | 0.072 |
| RandomForest | bagging | 0.940 | 0.897 | 0.058 |
| HistGradientBoosting | boosting | 0.939 | 0.890 | 0.067 |
| MLP (64, 32) | neural network | 0.938 | 0.896 | 0.061 |

HistGradientBoosting was chosen for tuning: it is within one standard deviation of RandomForest on CV, trains about 5x faster (3.3 s vs 17.7 s), predicts about 1.7x faster and the model is about 100x smaller.

## Bayesian optimisation (`results/bayes_opt_summary.json`, `results/bayes_opt_trials.csv`)
- Optuna TPE sampler, 20 trials, objective = grouped CV macro F1.
- Best parameters: learning_rate 0.073, max_leaf_nodes 114, max_depth 16, min_samples_leaf 11, l2_regularization 0.016, max_features 0.48.

| | CV macro F1 | Student-test Top-1 | Student-test macro F1 | Student-test log loss |
|---|---|---|---|---|
| HGB untuned | 0.9390 | 0.8901 | 0.9391 | 0.0674 |
| HGB tuned | 0.9426 | 0.9051 | 0.9474 | 0.0564 |

## Over/underfitting
Train vs CV macro F1 gaps are small (about 0.002 to 0.022). Student-test scores are close to CV scores, so the grouped CV estimate generalises.
