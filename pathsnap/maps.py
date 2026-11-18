"""Folium map builder: raw GPS, road network, reference route and predicted routes as layers."""
import numpy as np
import folium
from branca.element import Element


def _segments(trip, arcs):
    return [[(trip.node_lat[trip.u[a]], trip.node_lon[trip.u[a]]), (trip.node_lat[trip.v[a]], trip.node_lon[trip.v[a]])] for a in arcs]


def build_map(trip, ref_arcs=None, pred_layers=(), title="", margin_m=250, focus=None):
    lat_c, lon_c = float(trip.track.lat.median()), float(trip.track.lon.median())
    m = folium.Map(location=[lat_c, lon_c], zoom_start=14, tiles="OpenStreetMap", control_scale=True)
    # road network near the track (single multi-line layer keeps the file small)
    lo_x, hi_x = trip.gx.min() - margin_m, trip.gx.max() + margin_m
    lo_y, hi_y = trip.gy.min() - margin_m, trip.gy.max() + margin_m
    ins = (trip.nx >= lo_x) & (trip.nx <= hi_x) & (trip.ny >= lo_y) & (trip.ny <= hi_y)
    sel = np.where(trip.valid_arc & ins[trip.u] & ins[trip.v] & (trip.u < trip.v) | (trip.valid_arc & ins[trip.u] & ins[trip.v] & (trip.two_way == 0)))[0]
    folium.PolyLine(_segments(trip, sel), color="#8a8a8a", weight=1.2, opacity=0.6).add_to(
        folium.FeatureGroup("Road network (near track)", show=True).add_to(m))
    if ref_arcs is not None:
        fg = folium.FeatureGroup("Reference route", show=True).add_to(m)
        folium.PolyLine(_segments(trip, np.unique(ref_arcs)), color="#1a9641", weight=7, opacity=0.55).add_to(fg)
    fg = folium.FeatureGroup("Raw GPS observations", show=True).add_to(m)
    step = max(1, len(trip.track) // 1500)
    for la, lo, t in trip.track[["lat", "lon", "t"]].values[::step]:
        folium.CircleMarker((la, lo), radius=2, color="#2b83ba", fill=True, fill_opacity=0.9, weight=1, tooltip=f"t={t:.0f}s").add_to(fg)
    colors = ["#ff7f00", "#d7191c", "#7b3294"]
    for (name, arcs), col in zip(pred_layers, colors):
        fg = folium.FeatureGroup(name, show=True).add_to(m)
        seq = arcs[np.r_[True, arcs[1:] != arcs[:-1]]]
        folium.PolyLine(_segments(trip, seq), color=col, weight=3.5, opacity=0.9).add_to(fg)
    pts = trip.track if focus is None else trip.track.iloc[focus[0]:focus[1]]
    m.fit_bounds([[pts.lat.min(), pts.lon.min()], [pts.lat.max(), pts.lon.max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    rows = "".join(f'<div><span style="display:inline-block;width:22px;height:5px;background:{c};margin-right:6px;vertical-align:middle"></span>{n}</div>'
                   for n, c in [("Road network", "#8a8a8a")] + ([("Reference route", "#1a9641")] if ref_arcs is not None else [])
                   + [(n, c) for (n, _), c in zip(pred_layers, colors)])
    rows += '<div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#2b83ba;margin:0 13px 0 6px"></span>Raw GPS</div>'
    legend = f'''<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 12px;border-radius:6px;
      box-shadow:0 1px 5px rgba(0,0,0,.3);font:12px sans-serif;line-height:1.7"><b>{title}</b>{rows}</div>'''
    m.get_root().html.add_child(Element(legend))
    return m
