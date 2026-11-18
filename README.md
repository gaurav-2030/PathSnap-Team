# PathSnap

GPS map-matching: reconstruct the road route driven by a vehicle from a noisy GPS track and a road network.
Pipeline: candidate generation, candidate classifier (Bayesian-tuned HistGradientBoosting), HMM / Viterbi decoding, FastAPI service and web UI.

## Team and contributions

| Member | Area | Main files |
|---|---|---|
| Gaurav Girish (PES1UG24AM103) | Data exploration, documentation | `scripts/01_explore_dataset.py`, `docs/DATA_EXPLORATION.md` |
| Gonu Guntla Hema Kowshik (PES1UG24AM106) | Model experimentation and Bayesian optimisation | `scripts/03_train_models.py`, `scripts/04_bayes_opt.py`, `scripts/08_results_summary.py`, `docs/MODELS_AND_TUNING.md` |
| Gaurav Shivakumar (PES1UG24AM104) | FastAPI service, UI and two-laptop deployment | `api/main.py`, `ui/index.html`, `scripts/check_api.py`, `docs/DEPLOYMENT.md` |
| Gayathri Priya V (PES1UG24AM105) | Candidate generation, HMM/Viterbi decoder, tests | `pathsnap/core.py`, `scripts/05_hmm_viterbi.py`, `tests/test_core.py`, `docs/HMM_VITERBI.md` |

## Documentation
- [Data exploration](docs/DATA_EXPLORATION.md)
- [Models and Bayesian optimisation](docs/MODELS_AND_TUNING.md)
- [HMM / Viterbi decoder](docs/HMM_VITERBI.md)
- [Deployment on two laptops](docs/DEPLOYMENT.md)

## Repository layout

| Folder | Contents |
|---|---|
| `pathsnap/` | Library code: raw-file readers (`data_io.py`), geometry, candidate generation, route alignment, features, network distances, Viterbi (`core.py`), metrics (`evaluation.py`), inference pipeline (`pipeline.py`), Folium maps (`maps.py`) |
| `scripts/` | Numbered, reproducible experiment scripts (see below) |
| `api/` | FastAPI backend (`main.py`) |
| `ui/` | Evaluator interface (`index.html`, static page using Leaflet) |
| `artifacts/` | Trained models (`models/*.joblib`), final deployment config (`final_config.json`), Optuna study (`optuna_hgb.db`) |
| `results/` | CSV results, experiment logs, figures (`figures/`), Folium maps (`maps/`) |
| `report/` | Technical report (PDF) and its generator |
| `data/raw/` | Raw trip folders (not committed, see below) |

## Data

Place the trip folders as follows (each trip folder holds `<id>.track`, `<id>.nodes`, `<id>.arcs`, `<id>.route`):

```
data/raw/train/<trip_id>/            80 trips   – fitting, cross-validation, optimisation
data/raw/student_test/<trip_id>/     10 trips   – development-stage comparison
data/raw/evaluator_test/<trip_id>/   10 trips   – final evaluation only (used by the API, never by training)
```

## Installation

Python 3.11+ (developed with 3.13).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the experiments

Run from the repository root, in order:

```bash
python scripts/01_explore_dataset.py   # EDA tables, plots, Folium maps (train + student_test)
python scripts/02_build_dataset.py     # candidates, reference alignment, features -> data/processed/
python scripts/03_train_models.py      # 7 models, 5-fold GroupKFold CV + student-test evaluation
python scripts/04_bayes_opt.py         # Optuna TPE search (30 trials) + tuned vs untuned comparison
python scripts/05_hmm_viterbi.py       # decoder tuning, classifier vs classifier+HMM, final_config.json
python scripts/06_route_maps.py        # Folium maps of predicted routes
python scripts/07_plots.py             # comparison figures
```

All random seeds are fixed to 42. Logs of the runs are in `results/logs/`.

## Deployment (two laptops)

**Laptop 1 – model + API**

1. Copy the 10 evaluator trips into `data/raw/evaluator_test/` (or set `PATHSNAP_TRIPS_DIR` to their folder).
2. Start the API on all interfaces:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

3. Find the laptop's IP address (`ipconfig` on Windows, `ipconfig getifaddr en0` on macOS) and allow port 8000 through the firewall.
   Every request is logged in the terminal and in `logs/api_requests.log`.

Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status and loaded model |
| GET | `/trips` | list of evaluation trip ids |
| POST | `/predict` `{"trip_id": "00000015"}` | runs the full pipeline, returns the predicted route and server-side accuracy |

**Laptop 2 – UI**

```bash
cd ui
python -m http.server 5500
```

Open `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000`, or type the API address in the header and press **Load trips**. Select any trip: the UI sends only the trip id; the API predicts the route, scores it against its internal reference and returns the result. The UI draws the GPS points and the predicted route and shows the metrics.

Quick check from any machine:

```bash
curl -X POST http://<LAPTOP1_IP>:8000/predict -H "Content-Type: application/json" -d '{"trip_id": "00000015"}'
```

## Leakage safeguards

- Training, cross-validation, model selection, Bayesian optimisation and decoder tuning use only `train` (and `student_test` for development comparison).
- The predictor (`pathsnap/pipeline.py`) refuses input containing a reference route; the API reads the `.route` file only after prediction, to compute accuracy, and never returns it.
- No trip ids or predictions are hard-coded anywhere.
