"""Observation-level and route-level evaluation helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix

from .core import Trip, continuity, route_set_scores
from .data_io import load_trip


def group_bounds(keys):
    """Start/end row indices of consecutive equal keys (rows are sorted by trip, obs)."""
    chg = np.r_[True, keys[1:] != keys[:-1]]
    starts = np.flatnonzero(chg)
    return starts, np.r_[starts[1:], len(keys)]


def obs_keys(df):
    return df.trip.astype(str).values + ":" + df.obs.astype(str).values


def top1_mask(df, score):
    """1 for the highest-scored candidate of every observation, else 0."""
    keys = df["_key"].values if "_key" in df else obs_keys(df)
    starts, ends = group_bounds(keys)
    pick = np.zeros(len(df), dtype=np.int8)
    for s, e in zip(starts, ends):
        pick[s + int(np.argmax(score[s:e]))] = 1
    return pick


def observation_metrics(df, prob):
    """Top-1 accuracy, macro P/R/F1 (candidate level, top-1 decision) and log loss."""
    y = df.y.values
    pick = top1_mask(df, prob)
    keys = df["_key"].values if "_key" in df else obs_keys(df)
    starts, _ = group_bounds(keys)
    n_obs = len(starts)
    top1 = float((pick & y).sum() / n_obs)
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return {
        "top1": top1,
        "macro_f1": f1_score(y, pick, average="macro"),
        "macro_precision": precision_score(y, pick, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pick, average="macro", zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "cand_miss_rate": float(1 - np.maximum.reduceat(y, starts).mean()),
        "n_obs": n_obs,
    }


def confusion(df, prob):
    return confusion_matrix(df.y.values, top1_mask(df, prob), labels=[0, 1])


class RouteEvaluator:
    """Caches Trip objects of a partition to score route continuity / route-set overlap."""

    def __init__(self, partition, trips):
        self.partition = partition
        self.trips = {t: Trip(load_trip(partition, t), t) for t in trips}
        self.ref_cache = {}

    def predicted_arcs(self, df, pick):
        out = {}
        sel = df[pick.astype(bool)]
        for t, g in sel.groupby("trip"):
            out[t] = (g.obs.values, g.arc.values)
        return out

    def scores(self, df, pick):
        rows = []
        for t, (obs, arcs) in self.predicted_arcs(df, pick).items():
            trip = self.trips[t]
            cont = continuity(trip, arcs, obs)
            ref = trip.route
            seq = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
            prec, rec, f1 = route_set_scores(seq, ref)
            rows.append({"trip": t, "continuity": cont, "route_precision": prec,
                         "route_recall": rec, "route_f1": f1})
        return pd.DataFrame(rows)

    def reference_continuity(self, df):
        rows = []
        for t, g in df[df.y == 1].groupby("trip"):
            trip = self.trips[t]
            rows.append({"trip": t, "continuity": continuity(trip, g.arc.values, g.obs.values)})
        return pd.DataFrame(rows)


def metrics_from_pick(df, pick):
    """Observation metrics for a hard selection (one chosen candidate per observation)."""
    y = df.y.values
    keys = df["_key"].values if "_key" in df else obs_keys(df)
    starts, _ = group_bounds(keys)
    return {
        "top1": float((pick & y).sum() / len(starts)),
        "macro_f1": f1_score(y, pick, average="macro"),
        "macro_precision": precision_score(y, pick, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pick, average="macro", zero_division=0),
        "n_obs": len(starts),
    }
