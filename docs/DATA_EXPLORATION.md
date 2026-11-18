# Data exploration

Script: `scripts/01_explore_dataset.py`. Outputs: `results/eda_*.csv`, `results/eda/`, `results/figures/eda_*.png`, `results/maps/eda_*.html`.
Only the train (80 trips) and student-test (10 trips) partitions were explored. The evaluator partition was never used.

## Dataset structure
Each trip folder has four header-less, whitespace-separated files:

| File | Columns | Meaning |
|---|---|---|
| `.track` | longitude, latitude, time (s) | GPS observations |
| `.nodes` | longitude, latitude | road nodes (row index = node id) |
| `.arcs` | from-node, to-node | directed straight road segments (row index = arc id) |
| `.route` | arc id | ordered reference route |

Column order was verified: every track lies inside its node bounding box, and all arc and route indices are valid.

## Size
| | Train | Student-test |
|---|---|---|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| GPS points per trip (mean / min / max) | 2605 / 361 / 19313 | 1916 / 262 / 3958 |
| Road arcs per trip (mean) | 48,461 | 18,504 |
| Reference-route arcs per trip (mean) | 392 | 348 |
| Track length per trip (mean km) | 37.2 | 29.7 |

## Missing values
Every line of every file was parsed. No missing files, no unparseable rows, no NaN coordinates, no duplicate GPS rows, no non-increasing timestamps (`results/eda_missing_values.csv`).

## Findings and what we did about them
| Finding | Evidence | Decision |
|---|---|---|
| Sampling is mostly 1 Hz but irregular | 96.5% of steps are exactly 1 s; 590 gaps > 10 s in train (largest 153,064 s, trip 90) | HMM transition tolerance grows with the time step; continuity is not enforced across gaps > 60 s |
| GPS spikes | implied speeds up to 648 km/h (trip 79) | context features over neighbouring points + HMM smoothing |
| Two-way roads | 92% of arcs have an opposite twin with identical geometry; nearest-arc rule = 54.3% Top-1 | heading features |
| Dense, partly disconnected networks | about 39 weakly connected components per trip; median arc 16 to 94 m | shortest-path transitions, chain restarts |
| Duplicates / self-loops | 1,775 duplicate arcs, 22 self-loops (train) | duplicates merged, self-loops excluded |
| Imperfect reference route | 11 trips with route breaks; trip 56 revisits arcs 255 times; trips 33 and 54 run about 140 m from the route | points > 50 m from their aligned arc (11.2%) excluded from training |
| Worldwide coordinates | longitude -100.4 to 137.2 | local equirectangular projection to metres per trip |

## Figures
- `eda_distributions.png`: GPS points, track length, network size and location per trip
- `eda_sampling_intervals.png`: time step distribution (train)
- `eda_trajectories.png`: GPS, reference route and road network in local metres
- `eda_gps_to_route.png`: per-trip median GPS-to-route distance
- `results/maps/eda_*.html`: interactive Folium maps with layer control
