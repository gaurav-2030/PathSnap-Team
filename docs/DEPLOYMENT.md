# Deployment on two laptops

PathSnap runs as two parts that talk over HTTP on the same Wi-Fi or hotspot:

| Laptop | Runs | Port |
|---|---|---|
| Laptop 1 (Windows) | FastAPI backend: loads the tuned model, generates candidates, scores them, runs Viterbi | 8000 |
| Laptop 2 (macOS) | Evaluator UI: static page `ui/index.html` (HTML, JavaScript, Leaflet map) | 5500 |

## How a request flows

1. The evaluator picks a trip in the UI on Laptop 2.
2. The browser sends `POST http://<LAPTOP1_IP>:8000/predict` with the body `{"trip_id": "00000015"}`. Only the trip id is sent.
3. The API reads that trip's track, nodes and arcs from disk, runs candidate generation, the classifier and HMM/Viterbi decoding, and only then reads the `.route` file to compute accuracy.
4. The API returns JSON with the predicted route geometry, GPS points (for drawing) and the metrics. The reference route is never returned.
5. The UI draws the predicted route (red) over the GPS points (blue) and shows Top-1 accuracy (HMM and classifier only), route precision, recall and F1, and continuity.

CORS is enabled in `api/main.py` so the browser on Laptop 2 can call the API on Laptop 1. Every request is logged in the API terminal and in `logs/api_requests.log`.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status, loaded model, number of trips |
| GET | `/trips` | list of trip ids in the trips folder |
| POST | `/predict` `{"trip_id": "..."}` | predicted route, GPS points, metrics, continuity, timing |

## Laptop 1: Windows (PowerShell)

Requires Python 3.13 (python.org, with "Add python.exe to PATH" ticked).

```powershell
git clone https://github.com/gaurav-2030/PathSnap-Team.git
cd PathSnap-Team
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy the trip folders (each holds `<id>.track`, `<id>.nodes`, `<id>.arcs`, `<id>.route`):

- Evaluator trips go in `data\raw\evaluator_test\<trip_id>\`. This is the default folder the API reads.
- For a rehearsal with the student-test trips, put them in `data\raw\student_test\` and point the API at them:

```powershell
$env:PATHSNAP_TRIPS_DIR = "$PWD\data\raw\student_test"
```

Start the API on all network interfaces:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

When Windows Firewall asks, allow access on both Private and Public networks. If there is no prompt and Laptop 2 cannot connect, run this once in an Administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "PathSnap 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

Find the IP address with `ipconfig` (IPv4 Address under "Wireless LAN adapter Wi-Fi"). Check locally:

```powershell
curl.exe http://localhost:8000/health
python scripts\check_api.py
```

To switch back from student-test to evaluator trips, stop the API (Ctrl+C), run `Remove-Item Env:PATHSNAP_TRIPS_DIR`, and start it again.

## Laptop 2: macOS (Terminal)

```bash
cd PathSnap-Team/ui
python3 -m http.server 5500
```

Open `http://localhost:5500/?api=http://<LAPTOP1_IP>:8000` in the browser. Alternatively, open the page, type the API address in the header box and press **Load trips**.

Check the connection from the Mac:

```bash
curl http://<LAPTOP1_IP>:8000/health
python3 scripts/check_api.py --api http://<LAPTOP1_IP>:8000 --all
```

## Troubleshooting

| Problem | Fix |
|---|---|
| UI cannot connect | Wrong IP, different network, firewall, or the API is stopped. Open `http://<LAPTOP1_IP>:8000/health` in Laptop 2's browser. |
| Port 8000 in use (Windows) | `netstat -ano \| findstr :8000`, then `taskkill /PID <pid> /F` |
| Port 5500 in use (macOS) | `lsof -ti:5500 \| xargs kill`, or use another port |
| `Activate.ps1` is blocked | Run the `Set-ExecutionPolicy -Scope Process ...` line again in the same window |
| `/trips` is empty | No trip folders in `data/raw/evaluator_test/`, or `PATHSNAP_TRIPS_DIR` points to the wrong folder |

## Single-machine fallback

Both parts can run on one laptop: start the API with `--port 8000` and open `http://localhost:5500/?api=http://localhost:8000`.
