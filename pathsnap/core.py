"""PathSnap core: geometry, candidate generation, label alignment, features,
network distances, Viterbi decoding and route metrics."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

EARTH_R = 6371008.8

# defaults (candidate generation is fixed before model comparison; see report section 2)
CAND_RADIUS = 50.0      # metres
CAND_K = 20             # max candidates per observation
MIN_CAND = 2            # always keep at least this many nearest arcs
SAMPLE_STEP = 10.0      # densification step for the arc index (m)
CROP_MARGIN = 1500.0    # graph crop margin around the track (m)

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cand", "cand_frac_closer",
    "head_cos", "head_valid", "speed", "proj_t", "off_end", "arc_len",
    "ctx_dist_mean", "ctx_dist_max", "d_prev", "d_next", "ctx_head_cos",
    "out_deg_v", "in_deg_u", "two_way", "dt_prev", "dt_next",
]


# ---------------------------------------------------------------- geometry
def project(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres around (lon0, lat0)."""
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y


def point_seg(px, py, ax, ay, bx, by):
    """Distance from points to segments (all arrays broadcastable).
    Returns distance, clipped projection fraction and unclipped fraction."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t_raw = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(l2 > 0, l2, 1), 0.0)
    t = np.clip(t_raw, 0.0, 1.0)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(px - qx, py - qy), t, t_raw


# ---------------------------------------------------------------- trip preparation
class Trip:
    """Holds one trip in projected metres with a cleaned arc table."""

    def __init__(self, raw, tid=""):
        self.tid = tid
        trk = raw["track"].dropna().copy()
        trk = trk[(trk.lat.abs() <= 90) & (trk.lon.abs() <= 180)]
        trk = trk.sort_values("t", kind="stable").drop_duplicates()
        self.track = trk.reset_index(drop=True)
        self.lon0 = float(self.track.lon.mean())
        self.lat0 = float(self.track.lat.mean())
        self.gx, self.gy = project(self.track.lon.values, self.track.lat.values, self.lon0, self.lat0)
        self.t = self.track.t.values.astype(float)

        nodes = raw["nodes"]
        self.node_lon, self.node_lat = nodes.lon.values, nodes.lat.values
        self.nx, self.ny = project(self.node_lon, self.node_lat, self.lon0, self.lat0)
        arcs = raw["arcs"]
        self.u = arcs.u.values.astype(np.int64)
        self.v = arcs.v.values.astype(np.int64)
        self.n_arcs = len(self.u)
        self.arc_len = np.hypot(self.nx[self.v] - self.nx[self.u], self.ny[self.v] - self.ny[self.u])

        # duplicate (u, v) rows describe the same directed road piece: map to the lowest arc id
        key = self.u * (len(self.nx) + 1) + self.v
        first = pd.Series(np.arange(self.n_arcs)).groupby(key).transform("min").values
        self.canon = first.astype(np.int64)
        self.valid_arc = (self.canon == np.arange(self.n_arcs)) & (self.u != self.v)

        n_nodes = len(self.nx)
        self.out_deg = np.bincount(self.u[self.valid_arc], minlength=n_nodes)
        self.in_deg = np.bincount(self.v[self.valid_arc], minlength=n_nodes)
        rev = set(zip(self.v[self.valid_arc].tolist(), self.u[self.valid_arc].tolist()))
        self.two_way = np.array([(a, b) in rev for a, b in zip(self.u.tolist(), self.v.tolist())], dtype=np.int8)

        self.route = None
        if raw.get("route") is not None:
            self.route = self.canon[raw["route"].arc_id.values.astype(np.int64)]
        self._build_index()
        self._graph = None

    # spatial index over densified arcs
    def _build_index(self):
        ids = np.where(self.valid_arc)[0]
        lo_x, hi_x = self.gx.min() - 300, self.gx.max() + 300
        lo_y, hi_y = self.gy.min() - 300, self.gy.max() + 300
        ax, ay, bx, by = self.nx[self.u[ids]], self.ny[self.u[ids]], self.nx[self.v[ids]], self.ny[self.v[ids]]
        near = ~((np.maximum(ax, bx) < lo_x) | (np.minimum(ax, bx) > hi_x) |
                 (np.maximum(ay, by) < lo_y) | (np.minimum(ay, by) > hi_y))
        ids = ids[near]
        n_s = np.maximum(1, np.ceil(self.arc_len[ids] / SAMPLE_STEP).astype(int)) + 1
        rep = np.repeat(ids, n_s)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n_s])
        sx = self.nx[self.u[rep]] + frac * (self.nx[self.v[rep]] - self.nx[self.u[rep]])
        sy = self.ny[self.u[rep]] + frac * (self.ny[self.v[rep]] - self.ny[self.u[rep]])
        self.idx_arc = rep
        self.tree = cKDTree(np.c_[sx, sy])

    # ------------------------------------------------------------ candidates
    def candidates(self, radius=CAND_RADIUS, k=CAND_K, min_cand=MIN_CAND):
        """Return DataFrame (obs, arc, dist, t, t_raw) with up to k arcs per observation."""
        pts = np.c_[self.gx, self.gy]
        hits = self.tree.query_ball_point(pts, r=radius + SAMPLE_STEP)
        _, nn = self.tree.query(pts, k=min(64, len(self.idx_arc)))
        obs_l, arc_l = [], []
        for i, h in enumerate(hits):
            arcs = np.unique(self.idx_arc[np.asarray(h, dtype=int)]) if len(h) else np.empty(0, int)
            extra = np.unique(self.idx_arc[nn[i]])
            arcs = np.union1d(arcs, extra)
            obs_l.append(np.full(len(arcs), i)); arc_l.append(arcs)
        obs = np.concatenate(obs_l); arc = np.concatenate(arc_l)
        d, t, t_raw = point_seg(self.gx[obs], self.gy[obs], self.nx[self.u[arc]], self.ny[self.u[arc]],
                                self.nx[self.v[arc]], self.ny[self.v[arc]])
        df = pd.DataFrame({"obs": obs, "arc": arc, "dist": d, "t": t, "t_raw": t_raw})
        df = df.sort_values(["obs", "dist", "arc"], kind="stable")
        df["dist_rank"] = df.groupby("obs").cumcount()
        keep = (df.dist_rank < min_cand) | ((df.dist <= radius) & (df.dist_rank < k))
        return df[keep].reset_index(drop=True)

    # ------------------------------------------------------------ reference labels
    def align_route(self):
        """Per-observation reference arc from the ordered route via a monotone
        min-cost alignment (route order is preserved, every point gets one route position)."""
        r = self.route
        a, b = self.u[r], self.v[r]
        n, m = len(self.gx), len(r)
        lab_pos = np.empty(n, dtype=np.int64)
        back = np.empty((n, m), dtype=np.int32)
        prev = None
        for i in range(n):
            c, _, _ = point_seg(self.gx[i], self.gy[i], self.nx[a], self.ny[a], self.nx[b], self.ny[b])
            if prev is None:
                cur = c
                back[i] = np.arange(m)
            else:
                cm = np.minimum.accumulate(prev)
                # index of running minimum
                idx = np.arange(m)
                is_new = np.r_[True, prev[1:] < cm[:-1]]
                arg = np.maximum.accumulate(np.where(is_new, idx, 0))
                cur = c + cm
                back[i] = arg
            prev = cur
        j = int(np.argmin(prev))
        for i in range(n - 1, -1, -1):
            lab_pos[i] = j
            j = back[i, j]
        lab = r[lab_pos]
        dist, _, _ = point_seg(self.gx, self.gy, self.nx[self.u[lab]], self.ny[self.u[lab]],
                               self.nx[self.v[lab]], self.ny[self.v[lab]])
        return lab, lab_pos, dist

    # ------------------------------------------------------------ features
    def kinematics(self, w=2):
        n = len(self.gx)
        i0 = np.clip(np.arange(n) - w, 0, n - 1); i1 = np.clip(np.arange(n) + w, 0, n - 1)
        hx, hy = self.gx[i1] - self.gx[i0], self.gy[i1] - self.gy[i0]
        disp = np.hypot(hx, hy)
        dtw = np.maximum(self.t[i1] - self.t[i0], 1e-6)
        speed = disp / dtw
        valid = disp > 4.0
        dt_prev = np.r_[0.0, np.diff(self.t)]
        dt_next = np.r_[np.diff(self.t), 0.0]
        return hx, hy, disp, speed, valid, dt_prev, dt_next

    def features(self, cand):
        c = cand.copy()
        n = len(self.gx)
        o = c.obs.values; a = c.arc.values
        ax, ay = self.nx[self.u[a]], self.ny[self.u[a]]
        bx, by = self.nx[self.v[a]], self.ny[self.v[a]]
        g = c.groupby("obs")
        dmin = g.dist.transform("min").values
        c["dist_minus_min"] = c.dist.values - dmin
        c["dist_ratio"] = c.dist.values / (dmin + 1.0)
        c["n_cand"] = g.dist.transform("size").values
        c["cand_frac_closer"] = c.dist_rank.values / np.maximum(c.n_cand.values - 1, 1)
        hx, hy, disp, speed, valid, dt_prev, dt_next = self.kinematics()
        ex, ey = bx - ax, by - ay
        el = np.maximum(np.hypot(ex, ey), 1e-6)
        hd = np.maximum(disp[o], 1e-6)
        hc = (hx[o] * ex + hy[o] * ey) / (hd * el)
        c["head_cos"] = np.where(valid[o], hc, 0.0)
        c["head_valid"] = valid[o].astype(np.int8)
        c["speed"] = speed[o]
        c["proj_t"] = c.t.values
        c["off_end"] = np.maximum(0, np.maximum(-c.t_raw.values, c.t_raw.values - 1)) * el
        c["arc_len"] = el
        ctx = []
        for k in (-3, -2, -1, 1, 2, 3):
            j = np.clip(o + k, 0, n - 1)
            dk, _, _ = point_seg(self.gx[j], self.gy[j], ax, ay, bx, by)
            ctx.append(dk)
        ctx = np.vstack(ctx)
        c["ctx_dist_mean"] = ctx.mean(0)
        c["ctx_dist_max"] = ctx.max(0)
        c["d_prev"] = ctx[2]
        c["d_next"] = ctx[3]
        # heading of a wider window (smooths noise at low speed)
        hx5, hy5, disp5, _, valid5, _, _ = self.kinematics(w=5)
        hc5 = (hx5[o] * ex + hy5[o] * ey) / (np.maximum(disp5[o], 1e-6) * el)
        c["ctx_head_cos"] = np.where(valid5[o], hc5, 0.0)
        c["out_deg_v"] = self.out_deg[self.v[a]]
        c["in_deg_u"] = self.in_deg[self.u[a]]
        c["two_way"] = self.two_way[a]
        c["dt_prev"] = np.minimum(dt_prev[o], 600)
        c["dt_next"] = np.minimum(dt_next[o], 600)
        return c

    # ------------------------------------------------------------ network distances
    def graph(self):
        if self._graph is None:
            lo_x, hi_x = self.gx.min() - CROP_MARGIN, self.gx.max() + CROP_MARGIN
            lo_y, hi_y = self.gy.min() - CROP_MARGIN, self.gy.max() + CROP_MARGIN
            inside = (self.nx >= lo_x) & (self.nx <= hi_x) & (self.ny >= lo_y) & (self.ny <= hi_y)
            m = self.valid_arc & inside[self.u] & inside[self.v]
            nodes = np.where(inside)[0]
            remap = -np.ones(len(self.nx), dtype=np.int64); remap[nodes] = np.arange(len(nodes))
            G = csr_matrix((np.maximum(self.arc_len[m], 0.01), (remap[self.u[m]], remap[self.v[m]])),
                           shape=(len(nodes), len(nodes)))
            self._graph = (G, remap)
        return self._graph

    def node_dist(self, src, dst, limit):
        """Shortest directed path length between node pairs (arrays), inf if not within limit."""
        G, remap = self.graph()
        s, d = remap[src], remap[dst]
        out = np.full(len(src), np.inf)
        ok = (s >= 0) & (d >= 0)
        same = ok & (src == dst)
        out[same] = 0.0
        need = ok & ~same
        if need.any():
            us = np.unique(s[need])
            for k in range(0, len(us), 256):
                chunk = us[k:k + 256]
                D = dijkstra(G, directed=True, indices=chunk, limit=limit)
                pos = {int(x): r for r, x in enumerate(chunk)}
                sel = np.where(need & np.isin(s, chunk))[0]
                rows = np.array([pos[int(x)] for x in s[sel]])
                out[sel] = D[rows, d[sel]]
        return out

    def route_dist(self, a1, t1, a2, t2, limit):
        """Driving distance from position t1 on arc a1 to position t2 on arc a2."""
        L1, L2 = self.arc_len[a1], self.arc_len[a2]
        res = (1 - t1) * L1 + self.node_dist(self.v[a1], self.u[a2], limit) + t2 * L2
        same = a1 == a2
        res = np.where(same, np.abs(t2 - t1) * L1, res)
        return res


# ---------------------------------------------------------------- decoding
def pair_table(trip, cand, limit=3000.0):
    """All candidate pairs of consecutive observations with their driving distance (rd),
    great-circle GPS step (gc) and time step (dt). Independent of decoder parameters."""
    obs = cand.obs.values
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]
    n_obs = len(starts)
    tab = {"starts": starts, "ends": ends, "n_obs": n_obs}
    if n_obs < 2:
        return tab
    arcs = cand.arc.values; tt = cand.t.values
    P1, P2, step = [], [], []
    for i in range(1, n_obs):
        r1 = np.arange(starts[i - 1], ends[i - 1]); r2 = np.arange(starts[i], ends[i])
        g1, g2 = np.meshgrid(r1, r2, indexing="ij")
        P1.append(g1.ravel()); P2.append(g2.ravel()); step.append(np.full(g1.size, i))
    P1 = np.concatenate(P1); P2 = np.concatenate(P2); step = np.concatenate(step)
    oi = obs[starts]
    gc = np.hypot(np.diff(trip.gx[oi]), np.diff(trip.gy[oi]))
    dt = np.diff(trip.t[oi])
    tab.update(P1=P1, P2=P2, step=step,
               rd=trip.route_dist(arcs[P1], tt[P1], arcs[P2], tt[P2], limit),
               gc=gc[step - 1], dt=dt[step - 1],
               bnd=np.r_[0, np.flatnonzero(np.diff(step)) + 1, len(step)])
    return tab


def log_transitions(tab, beta, gap_s):
    """log P(transition) = -|driving distance - GPS distance| / (beta * sqrt(dt)).
    Unreachable pairs get -inf; across a time gap longer than gap_s every pair is allowed."""
    b = beta * np.sqrt(np.maximum(tab["dt"], 1.0))
    logt = -np.abs(tab["rd"] - tab["gc"]) / b
    logt[~np.isfinite(tab["rd"])] = -np.inf
    logt[tab["dt"] > gap_s] = 0.0
    return logt


def viterbi(logp, tab, logt, emis_w=1.0):
    """Viterbi over candidate states (rows). Returns the chosen row per observation and
    the number of chain breaks (steps where no candidate pair was connected)."""
    starts, ends, n_obs = tab["starts"], tab["ends"], tab["n_obs"]
    best = np.empty(n_obs, dtype=np.int64)
    score = emis_w * logp[starts[0]:ends[0]].copy()
    if n_obs == 1:
        best[0] = starts[0] + int(np.argmax(score))
        return best, 0
    bnd = tab["bnd"]
    backs = [None] * n_obs
    breaks = 0
    for i in range(1, n_obs):
        n1 = ends[i - 1] - starts[i - 1]; n2 = ends[i] - starts[i]
        lt = logt[bnd[i - 1]:bnd[i]].reshape(n1, n2)
        tot = score[:, None] + lt
        bp = np.argmax(tot, axis=0)
        val = tot[bp, np.arange(n2)]
        if not np.isfinite(val).any():
            # disconnected step: restart the chain from the best state so far
            breaks += 1
            bp = np.full(n2, int(np.argmax(score)))
            val = np.full(n2, score.max())
        score = val + emis_w * logp[starts[i]:ends[i]]
        score -= score.max()
        backs[i] = bp
    j = int(np.argmax(score))
    for i in range(n_obs - 1, -1, -1):
        best[i] = starts[i] + j
        if i > 0:
            j = int(backs[i][j])
    return best, breaks


def argmax_rows(cand, score):
    obs = cand.obs.values
    starts = np.r_[0, np.flatnonzero(np.diff(obs)) + 1]
    ends = np.r_[starts[1:], len(obs)]
    return np.array([s + int(np.argmax(score[s:e])) for s, e in zip(starts, ends)])


# ---------------------------------------------------------------- metrics
def continuity(trip, arcs_seq, obs_idx, slack=1.5, add=60.0, limit=3000.0, gap_s=30.0):
    """Share of consecutive observation pairs whose predicted arcs form a feasible
    directed move (same arc, or reachable with driving distance <= slack*gc + add)."""
    a1, a2 = arcs_seq[:-1], arcs_seq[1:]
    gc = np.hypot(np.diff(trip.gx[obs_idx]), np.diff(trip.gy[obs_idx]))
    dt = np.diff(trip.t[obs_idx])
    m = dt <= gap_s
    if m.sum() == 0:
        return np.nan
    a1, a2, gc = a1[m], a2[m], gc[m]
    same = a1 == a2
    d = trip.node_dist(trip.v[a1], trip.u[a2], limit)
    rev_ok = (trip.u[a1] == trip.v[a2]) & (trip.v[a1] == trip.u[a2])  # U-turn on same street
    ok = same | rev_ok | (d <= slack * gc + add)
    return float(ok.mean())


def route_set_scores(pred_arcs, ref_arcs):
    p, r = set(pred_arcs.tolist()), set(ref_arcs.tolist())
    tp = len(p & r)
    prec = tp / max(len(p), 1); rec = tp / max(len(r), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, f1
