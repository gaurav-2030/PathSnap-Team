"""Raw-file readers for the PathSnap trip folders (.track, .nodes, .arcs, .route)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PARTITIONS = ["train", "student_test", "evaluator_test"]


def trip_ids(partition):
    d = RAW / partition
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def _read(path, names, dtype=float):
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, engine="python")
    return df


def load_trip(partition, tid, with_route=True):
    d = RAW / partition / tid
    trip = {
        "track": _read(d / f"{tid}.track", ["lon", "lat", "t"]),
        "nodes": _read(d / f"{tid}.nodes", ["lon", "lat"]),
        "arcs": _read(d / f"{tid}.arcs", ["u", "v"]),
    }
    if with_route:
        r = _read(d / f"{tid}.route", ["arc_id"])
        trip["route"] = r
    return trip
