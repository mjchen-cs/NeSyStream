"""Local server of the rating app: serves app/ and stimuli/, and appends every rating to data/ratings.csv.

python subjective/serve.py            then open http://localhost:8000 in Chrome (full screen, F11)
Each answer is written immediately (one CSV row per trial, plus data/<subject>.json at the end), so a crash or a
withdrawal never loses earlier answers. No name or student number is ever collected.
"""
import csv, json, os, sys, datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(ROOT, "data"); os.makedirs(DATA, exist_ok=True)
CSV = os.path.join(DATA, "ratings_valid.csv")   # valid data only; first-round raw data are in data/_archive
COLS = ["subject", "age_group", "gender", "viewing_freq", "vision", "phase", "trial", "file", "rating", "rt_s",
        "played_s", "watch_s", "complete", "time"]


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def do_GET(self):
        if self.path == "/subjects":                     # IDs already used (one ID = one person)
            used = sorted({r["subject"] for f in ("ratings.csv", "ratings_v2.csv", "ratings_valid.csv") if os.path.exists(os.path.join(DATA, f))
                           for r in csv.DictReader(open(os.path.join(DATA, f), encoding="utf-8-sig"))})
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps(used).encode()); return
        if self.path in ("/", "/index.html"):
            self.path = "/app/index.html"
        return super().do_GET()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        try:
            self._save(body)
        except Exception as e:                       # e.g. the CSV is open (locked) in Excel
            print("!! SAVE FAILED:", repr(e), flush=True)
            self.send_response(500); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"ok": False, "error": repr(e)}).encode()); return
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def _save(self, body):
        if self.path == "/rating":
            new = not os.path.exists(CSV)
            if not new:   # never append under a different header (columns would shift)
                hdr = open(CSV, encoding="utf-8-sig").readline().strip().split(",")
                assert hdr == COLS, f"{CSV} has another header {hdr}; move it away before running"
            with open(CSV, "a", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, COLS, extrasaction="ignore")
                if new: w.writeheader()
                w.writerow({**body, "time": datetime.datetime.now().isoformat(timespec="seconds")})
        elif self.path == "/session":
            json.dump(body, open(os.path.join(DATA, f"{body.get('subject', 'unknown')}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def log_message(self, *a):
        pass


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
print(f"rating app: http://localhost:{port}   data -> {CSV}  (close this file in Excel while testing)")
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
