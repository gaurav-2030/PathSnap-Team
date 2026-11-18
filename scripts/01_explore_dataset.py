"""Dataset exploration on the train and student_test partitions.
Outputs: results/eda_inventory.csv, results/eda_summary.csv, results/eda_missing_values.csv,
results/figures/eda_*.png, results/maps/eda_*.html"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pathsnap.data_io import RAW, ROOT, trip_ids, load_trip
from pathsnap.core import Trip, EARTH_R
from pathsnap.maps import build_map

OUT = ROOT / "results"; FIG = OUT / "figures"; MAPS = OUT / "maps"
for d in (FIG, MAPS):
    d.mkdir(parents=True, exist_ok=True)


def hav(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def raw_parse_check(path, ncols):
    """Rows that do not parse into ncols finite numbers."""
    n = bad = 0
    with open(path) as fh:
        for line in fh:
            n += 1
            parts = line.split()
            try:
                ok = len(parts) == ncols and all(np.isfinite(float(p)) for p in parts)
            except ValueError:
                ok = False
            bad += not ok
    return n, bad


rows, miss = [], []
for part in ["train", "student_test"]:
    for tid in trip_ids(part):
        d = RAW / part / tid
        for ext, nc in [("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)]:
            f = d / f"{tid}.{ext}"
            n, bad = raw_parse_check(f, nc) if f.exists() else (0, 0)
            miss.append({"partition": part, "trip": tid, "file": ext, "exists": f.exists(), "rows": n, "unparseable_rows": bad})
        tr = load_trip(part, tid)
        trk, nod, arc, rte = tr["track"], tr["nodes"], tr["arcs"], tr["route"]
        r = {"partition": part, "trip": tid}
        for ext in ["track", "nodes", "arcs", "route"]:
            f = d / f"{tid}.{ext}"
            r[f"size_{ext}_kb"] = round(f.stat().st_size / 1024, 1) if f.exists() else 0
        r.update(n_gps=len(trk), n_nodes=len(nod), n_arcs=len(arc), n_route=len(rte))
        r["gps_nan"] = int(trk.isna().any(axis=1).sum())
        lon, lat, t = trk.lon.values, trk.lat.values, trk.t.values
        r["gps_out_of_range"] = int(((np.abs(lat) > 90) | (np.abs(lon) > 180)).sum())
        dt = np.diff(t)
        r.update(dt_nonpos=int((dt <= 0).sum()), dt_eq1_frac=round(float((dt == 1).mean()), 4),
                 dt_max=float(dt.max()), gaps_gt10s=int((dt > 10).sum()), dup_rows=int(trk.duplicated().sum()))
        step = hav(lon[:-1], lat[:-1], lon[1:], lat[1:])
        r["track_len_km"] = round(step.sum() / 1000, 2)
        r["max_step_m"] = round(float(step.max()), 1)
        r["max_speed_kmh"] = round(float(np.nanmax(step / np.where(dt > 0, dt, np.nan)) * 3.6), 1)
        r["duration_s"] = float(t[-1] - t[0])
        r.update(lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max())
        r["track_inside_node_bbox"] = bool(nod.lon.min() <= np.median(lon) <= nod.lon.max() and nod.lat.min() <= np.median(lat) <= nod.lat.max())
        G = nx.DiGraph(); G.add_nodes_from(range(len(nod))); G.add_edges_from(zip(arc.u, arc.v))
        r["arc_bad_index"] = int(((arc.u < 0) | (arc.v < 0) | (arc.u >= len(nod)) | (arc.v >= len(nod))).sum())
        r["self_loops"] = int((arc.u == arc.v).sum())
        r["dup_arcs"] = int(arc.duplicated().sum())
        r["isolated_nodes"] = int(sum(1 for n in G.nodes if G.degree(n) == 0))
        wcc = list(nx.weakly_connected_components(G))
        r["n_weak_components"] = len(wcc)
        r["largest_wcc_frac"] = round(max(map(len, wcc)) / len(nod), 4)
        r["n_strong_components"] = nx.number_strongly_connected_components(G)
        rev = set(zip(arc.v, arc.u))
        r["two_way_frac"] = round(float(np.mean([(u, v) in rev for u, v in zip(arc.u, arc.v)])), 4)
        ra = rte.arc_id.values
        r["route_bad_index"] = int(((ra < 0) | (ra >= len(arc))).sum())
        cont = arc.v.values[ra[:-1]] == arc.u.values[ra[1:]]
        r["route_contiguous_frac"] = round(float(cont.mean()), 4)
        r["route_repeat_arcs"] = int(len(ra) - len(set(ra)))
        trip = Trip(tr, tid)
        _, _, adist = trip.align_route()
        r["gps_to_route_median_m"] = round(float(np.median(adist)), 2)
        r["gps_to_route_p95_m"] = round(float(np.percentile(adist, 95)), 2)
        r["gps_far_from_route_frac"] = round(float((adist > 50).mean()), 4)
        r["route_arc_len_median_m"] = round(float(np.median(trip.arc_len[trip.route])), 1)
        rows.append(r)
        print(part, tid, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "eda_inventory.csv", index=False)
mv = pd.DataFrame(miss)
mv.groupby(["partition", "file"]).agg(trips=("trip", "count"), files_present=("exists", "sum"), rows=("rows", "sum"),
                                      unparseable_rows=("unparseable_rows", "sum")).to_csv(OUT / "eda_missing_values.csv")
summ = df.groupby("partition").agg(
    trips=("trip", "count"), gps_total=("n_gps", "sum"), gps_mean=("n_gps", "mean"), gps_min=("n_gps", "min"), gps_max=("n_gps", "max"),
    nodes_mean=("n_nodes", "mean"), arcs_mean=("n_arcs", "mean"), route_mean=("n_route", "mean"),
    nan_rows=("gps_nan", "sum"), nonpos_dt=("dt_nonpos", "sum"), gaps_gt10s=("gaps_gt10s", "sum"), dup_rows=("dup_rows", "sum"),
    track_km_mean=("track_len_km", "mean"), isolated_nodes=("isolated_nodes", "sum"), self_loops=("self_loops", "sum"),
    dup_arcs=("dup_arcs", "sum"), wcc_mean=("n_weak_components", "mean"), two_way_mean=("two_way_frac", "mean"),
    route_noncontig_trips=("route_contiguous_frac", lambda s: int((s < 1).sum())),
    gps_to_route_median=("gps_to_route_median_m", "median")).round(3)
summ.to_csv(OUT / "eda_summary.csv")
print(summ.T)

# ---- plots
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
for p, c in [("train", "tab:blue"), ("student_test", "tab:orange")]:
    s = df[df.partition == p]
    ax[0, 0].hist(s.n_gps, bins=20, alpha=.6, label=p, color=c)
    ax[0, 1].hist(s.track_len_km, bins=20, alpha=.6, label=p, color=c)
    ax[1, 0].hist(np.log10(s.n_arcs), bins=20, alpha=.6, label=p, color=c)
    ax[1, 1].scatter(s.lon_min, s.lat_min, label=p, color=c, s=15)
ax[0, 0].set_title("GPS observations per trip"); ax[0, 0].set_xlabel("number of GPS points in the trip"); ax[0, 0].set_ylabel("number of trips")
ax[0, 1].set_title("Track length per trip"); ax[0, 1].set_xlabel("track length (km, haversine)"); ax[0, 1].set_ylabel("number of trips")
ax[1, 0].set_title("Road-network size per trip"); ax[1, 0].set_xlabel("log10(number of road arcs)  (3 = 1,000, 4 = 10,000, 5 = 100,000)"); ax[1, 0].set_ylabel("number of trips")
ax[1, 1].set_title("Trip locations (south-west corner of each track)"); ax[1, 1].set_xlabel("longitude (degrees)"); ax[1, 1].set_ylabel("latitude (degrees)")
for a in ax.flat:
    a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "eda_distributions.png", dpi=130); plt.close()

dts = np.concatenate([np.diff(load_trip("train", t, False)["track"].t.values) for t in trip_ids("train")])
vals, cnt = np.unique(np.clip(np.round(dts), 0, 15), return_counts=True)
plt.figure(figsize=(6, 4)); plt.bar(vals, cnt); plt.yscale("log")
plt.xlabel("time step between consecutive GPS points (s, clipped at 15)"); plt.ylabel("count (log)")
plt.title(f"Train sampling intervals: {np.mean(dts == 1) * 100:.1f}% exactly 1 s"); plt.tight_layout()
plt.savefig(FIG / "eda_sampling_intervals.png", dpi=130); plt.close()

plt.figure(figsize=(6, 4))
plt.hist(df.gps_to_route_median_m, bins=30, color="tab:green")
plt.xlabel("median GPS-to-reference-route distance per trip (m)"); plt.ylabel("trips")
plt.title("How far GPS points lie from the reference route"); plt.tight_layout()
plt.savefig(FIG / "eda_gps_to_route.png", dpi=130); plt.close()

fig, axs = plt.subplots(2, 3, figsize=(13, 8))
for a, (part, tid) in zip(axs.flat, [("train", "00000001"), ("train", "00000090"), ("train", "00000054"),
                                    ("student_test", "00000000"), ("student_test", "00000052"), ("student_test", "00000077")]):
    trip = Trip(load_trip(part, tid), tid)
    near = trip.valid_arc & (np.hypot(trip.nx[trip.u] - trip.gx.mean(), trip.ny[trip.u] - trip.gy.mean()) < 1.2 * np.ptp(trip.gx) + 500)
    for k in np.where(near)[0][::1]:
        a.plot([trip.nx[trip.u[k]], trip.nx[trip.v[k]]], [trip.ny[trip.u[k]], trip.ny[trip.v[k]]], color="#ccc", lw=.5)
    for k in trip.route:
        a.plot([trip.nx[trip.u[k]], trip.nx[trip.v[k]]], [trip.ny[trip.u[k]], trip.ny[trip.v[k]]], color="green", lw=2)
    a.scatter(trip.gx, trip.gy, s=1.5, color="blue")
    a.set_xlim(trip.gx.min() - 200, trip.gx.max() + 200); a.set_ylim(trip.gy.min() - 200, trip.gy.max() + 200)
    a.set_aspect("equal"); a.set_title(f"{part} {tid}", fontsize=9); a.tick_params(labelsize=7)
fig.suptitle("GPS (blue), reference route (green) and road network (grey) in local metres")
plt.tight_layout(); plt.savefig(FIG / "eda_trajectories.png", dpi=120); plt.close()

# ---- Folium maps
for part, tid in [("train", "00000001"), ("train", "00000054"), ("student_test", "00000009")]:
    trip = Trip(load_trip(part, tid), tid)
    build_map(trip, ref_arcs=trip.route, title=f"EDA: {part} trip {tid}").save(str(MAPS / f"eda_{part}_{tid}.html"))
print("done")
