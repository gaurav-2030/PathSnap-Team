# PathSnap

GPS map-matching: reconstruct the road route driven by a vehicle from a noisy GPS track and a road network.
Pipeline: candidate generation, candidate classifier (Bayesian-tuned HistGradientBoosting), HMM / Viterbi decoding, FastAPI service and web UI.

Full project code: https://github.com/gaurav-2030/PathSnap

## Team and contributions

| Member | Area | Main files |
|---|---|---|
| Gaurav Girish (PES1UG24AM103) | Data exploration, documentation | `scripts/01_explore_dataset.py`, `docs/DATA_EXPLORATION.md` |
| MEMBER_2_NAME (SRN) | Model experimentation and Bayesian optimisation | `scripts/03_train_models.py`, `scripts/04_bayes_opt.py`, `scripts/08_results_summary.py`, `docs/MODELS_AND_TUNING.md` |
| MEMBER_3_NAME (SRN) | FastAPI service, UI and two-laptop deployment | `api/main.py`, `ui/index.html`, `scripts/check_api.py`, `docs/DEPLOYMENT.md` |
| MEMBER_4_NAME (SRN) | Candidate generation, HMM/Viterbi decoder, tests | `pathsnap/core.py`, `scripts/05_hmm_viterbi.py`, `tests/test_core.py`, `docs/HMM_VITERBI.md` |

## Documentation
- [Data exploration](docs/DATA_EXPLORATION.md)
- [Models and Bayesian optimisation](docs/MODELS_AND_TUNING.md)
- [HMM / Viterbi decoder](docs/HMM_VITERBI.md)
- [Deployment on two laptops](docs/DEPLOYMENT.md)
