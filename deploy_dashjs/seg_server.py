"""Trace-shaped HTTP server for the dash.js deployment.

python seg_server.py PORT TRACE_FILE LOG_JSON
Serves www/. Media segments (*.m4s except init) are sent at the rate of the throughput trace (Mbps, payload ratio
0.95 as in the simulator) after one RTT (80 ms); the trace clock starts at the first media request and loops.
Every media request is logged (representation = level, segment number = chunk + 1, bytes, start, end).
"""
import json, os, re, sys, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np

PORT, TRACE, LOGF = int(sys.argv[1]), sys.argv[2], sys.argv[3]
WWW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "www")
arr = np.loadtxt(TRACE, ndmin=2); T = arr[:, 0] - arr[0, 0]; R = np.maximum(arr[:, 1], 1e-3)
PERIOD = T[-1] + (T[-1] - T[-2] if len(T) > 1 else 1.0)
STATE = {"t0": None, "log": [], "lock": threading.Lock()}
RTT, PAYLOAD, BLOCK = 0.08, 0.95, 8192
PAT = re.compile(r"chunk-(\d+)-(\d+)\.m4s")


def rate_bps(t):
    tt = t % PERIOD
    i = max(0, int(np.searchsorted(T, tt, side="right")) - 1)
    return R[i] * 1e6 / 8.0 * PAYLOAD          # bytes / s


class H(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **k):
        super().__init__(*a, directory=WWW, **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        m = PAT.search(self.path)
        if not m:
            return super().do_GET()
        path = os.path.join(WWW, self.path.lstrip("/").split("?")[0])
        data = open(path, "rb").read()
        with STATE["lock"]:
            if STATE["t0"] is None:
                STATE["t0"] = time.time()
        t_start = time.time()
        time.sleep(RTT)
        self.send_response(200); self.send_header("Content-Type", "video/iso.segment")
        self.send_header("Content-Length", str(len(data))); self.end_headers()
        sent, nxt = 0, time.time()
        while sent < len(data):
            blk = data[sent:sent + BLOCK]
            self.wfile.write(blk); sent += len(blk)
            nxt += len(blk) / rate_bps(time.time() - STATE["t0"])
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.time()
        self.wfile.flush()
        with STATE["lock"]:
            STATE["log"].append({"level": int(m.group(1)), "k": int(m.group(2)) - 1, "bytes": len(data),
                                 "start": t_start - STATE["t0"], "end": time.time() - STATE["t0"]})
            json.dump(STATE["log"], open(LOGF, "w"))


if __name__ == "__main__":
    print("segment server on", PORT, TRACE, flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
