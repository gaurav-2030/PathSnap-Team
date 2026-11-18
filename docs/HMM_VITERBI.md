# Candidate generation and HMM / Viterbi decoding

Code: `pathsnap/core.py` (candidates, features, transitions, Viterbi), `pathsnap/pipeline.py` (inference), `scripts/05_hmm_viterbi.py` (decoder tuning and comparison). Tests: `tests/test_core.py`.

## Candidate generation
- Each trip is projected to local metres (equirectangular around the track centre, R = 6,371,008.8 m).
- Arcs are densified every 10 m into a KD-tree. For each GPS point, up to K = 20 arcs within 50 m are kept (always at least the 2 nearest).
- Miss rate (correct arc not among candidates): 0.30% of reliable train observations, 0.15% on student-test.

## Hidden Markov model
- Hidden state: the candidate arc of each GPS observation. Observation: the GPS point.
- Emission: log of the classifier probability (tuned HistGradientBoosting), weighted by w.
- Transition between candidates of consecutive observations:
  `log P = -|driving distance - GPS distance| / (beta * sqrt(dt))`
  where driving distance is the shortest path on the cropped road graph (Dijkstra, limit 3 km).
  Unreachable pairs get -inf. Across a time gap longer than `gap_s`, every pair is allowed (continuity not enforced).
- Viterbi finds the most likely sequence. If no candidate pair between two observations is connected, the chain breaks and restarts from the best state so far (counted as a chain break).

## Decoder tuning (`results/hmm_decoder_grid.csv`, train out-of-fold)
Best: beta = 2 m, w = 1.0, gap = 60 s (train OOF Top-1 0.9506).

## Results on student-test (`results/hmm_comparison_student_test.csv`)
| Variant | Top-1 | Macro F1 | Continuity | Route F1 |
|---|---|---|---|---|
| Nearest candidate | 0.5435 | 0.7454 | 0.9036 | 0.5578 |
| Distance-only HMM (Gaussian emission) | 0.9156 | 0.9533 | 0.9998 | 0.9299 |
| Classifier only (tuned HGB) | 0.9051 | 0.9474 | 0.9798 | 0.9044 |
| Classifier + HMM / Viterbi | 0.9378 | 0.9657 | 0.9998 | 0.9410 |

The HMM improved Top-1 on all 10 student-test trips (by 1.2 to 5.9 points) with 0 chain breaks.
Log loss is not reported for the HMM, because Viterbi returns one best path, not calibrated probabilities.

## Limitation
Arcs crossed between two GPS points are not filled in, so route recall barely changes.
