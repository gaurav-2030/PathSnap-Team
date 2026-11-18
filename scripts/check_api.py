"""Check a running PathSnap API from any machine on the same network.

Usage:
    python scripts/check_api.py                              # http://localhost:8000, first trip
    python scripts/check_api.py --api http://192.168.1.20:8000
    python scripts/check_api.py --api http://192.168.1.20:8000 --trip 00000015
    python scripts/check_api.py --api http://192.168.1.20:8000 --all

Calls GET /health, GET /trips and POST /predict, and prints the server-side metrics.
Only the trip id is sent; the reference route never leaves the API.
"""
import argparse
import sys
import time

import requests


def fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


def predict(api, tid, timeout):
    t0 = time.time()
    r = requests.post(f"{api}/predict", json={"trip_id": tid}, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    m = out.get("metrics") or {}
    print(f"trip {tid}: obs={out['n_observations']} candidates={out['n_candidates']} "
          f"route_arcs={len(out['predicted_route_arcs'])} "
          f"top1_hmm={fmt(m.get('top1_accuracy_hmm'))} top1_clf={fmt(m.get('top1_accuracy_classifier_only'))} "
          f"route_f1={fmt(m.get('route_f1'))} continuity={out['route_continuity']:.4f} "
          f"breaks={out['chain_breaks']} time={time.time() - t0:.2f}s")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--trip", help="trip id to predict (default: first trip listed by the API)")
    ap.add_argument("--all", action="store_true", help="predict every trip and print the mean Top-1")
    ap.add_argument("--timeout", type=float, default=120, help="request timeout in seconds")
    args = ap.parse_args()
    api = args.api.rstrip("/")

    try:
        h = requests.get(f"{api}/health", timeout=10).json()
    except requests.RequestException as e:
        sys.exit(f"Cannot reach {api}: {e}\nCheck the IP address, that both machines are on the same network, "
                 "that the API is running, and that port 8000 is allowed through the firewall.")
    print(f"health: status={h['status']} model={h['model']} trips={h['n_trips']}")

    trips = requests.get(f"{api}/trips", timeout=10).json()["trips"]
    print(f"trips: {', '.join(trips) if trips else '(none)'}")
    if not trips:
        sys.exit("The API has no trips. Put trip folders in data/raw/evaluator_test/ or set PATHSNAP_TRIPS_DIR.")

    targets = trips if args.all else [args.trip or trips[0]]
    accs = []
    for tid in targets:
        m = predict(api, tid, args.timeout)
        if m.get("top1_accuracy_hmm") is not None:
            accs.append(m["top1_accuracy_hmm"])
    if args.all and accs:
        print(f"mean Top-1 (HMM) over {len(accs)} trips: {sum(accs) / len(accs):.4f}")


if __name__ == "__main__":
    main()
