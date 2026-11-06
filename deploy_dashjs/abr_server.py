"""ABR decision service for the dash.js deployment (policy as a service).

POST /decide {sid, method, k, buffer, last: {bytes, dur_s} | null}  ->  {level, l_max, ms}
Methods: nesy (conformal feasible set + 3-step rule-based planning), mpc (RobustMPC, 5-chunk, quality), bba.
State per session is kept in an ABREnv instance of the simulator, whose internals (buffer, throughput history,
previous level, chunk index) are overwritten with what the real player reports, so the decision code is exactly
the one evaluated in simulation.
"""
import json, os, sys, time, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nesystream.env.abr_env import ABREnv, EnvConfig, B_IN_MB
from nesystream.models.nsqoe import NSQoE
from nesystream.eval.baselines import RobustMPC, BBA

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "src", "playlist.json")))
W = json.load(open(os.path.join(HERE, "weights_mos.json")))
VIDEO = {k: np.asarray(P[k], np.float32) for k in ("sizes", "vmaf", "pqf_q", "bitrates_kbps")}
VIDEO["pqf_concepts"] = np.asarray(P["concepts"], np.float32)
for st in (10, 20):                       # clip boundaries of the 3-clip playlist: SceneCut (as in the simulator)
    VIDEO["pqf_concepts"][st, 2] = 1.0
VIDEO.update({"chunk_len": np.float32(P["chunk_len"]), "heights": np.asarray(P["heights"]), "name": "playlist"})
DUMMY_TRACE = [{"name": "dummy", "time": np.array([0.0, 1e6]), "mbps": np.array([5.0, 5.0])}]


def new_env():
    cfg = EnvConfig(quality_source="pqf", use_hard_rules=True, hard_tput="conformal", conf_alpha=0.05,
                    conf_gamma=0.01, conf_file=os.path.join(HERE, "ratios.json"), log_tput=True)
    e = ABREnv([VIDEO], DUMMY_TRACE, NSQoE(init_weights=W), cfg, seed=0)
    e.reset(0, 0, random_start=False)
    return e


def plan_scores(e, roots, H):
    """Identical to scripts/evaluate.py (Ours-plan<H>-all)."""
    L, N, k = e.L, e.N, e.k
    c = torch.tensor([float(e.ctx.get("c_h1", e.ctx["c_low"]))]); rtt, cl = e.cfg.rtt, e.chunk_len
    lv = torch.as_tensor(roots, dtype=torch.long); root = torch.arange(len(lv))
    buf = torch.full((len(lv),), float(e.buffer)); ret = torch.zeros(len(lv))
    prev_q = torch.full((len(lv),), float(e.prev_q if e.prev_q is not None else e.ctx["q_curve"][0]))
    for j in range(min(H, N - k)):
        kk = k + j
        qh = torch.as_tensor(e._quality_curve(kk), dtype=torch.float32)
        sz = torch.as_tensor(e.v["sizes"][kk], dtype=torch.float32)
        cc = torch.as_tensor(e._concepts(kk), dtype=torch.float32)
        if j > 0:
            ch = (lv[:, None] + torch.tensor([0, 1, -1])[None]).clamp(0, L - 1)
            lv, root, buf, ret, prev_q = (ch.reshape(-1), root.repeat_interleave(3), buf.repeat_interleave(3),
                                          ret.repeat_interleave(3), prev_q.repeat_interleave(3))
            dl = sz[lv] * 8.0 / c + rtt
            ok = ((buf - dl) >= e.cfg.b_min) | (lv == 0)
            lv, root, buf, ret, prev_q = lv[ok], root[ok], buf[ok], ret[ok], prev_q[ok]
        B = len(lv)
        with torch.no_grad():
            tab = e.q.step_qoe_all_levels(qh.expand(B, L), sz.expand(B, L), c.expand(B), buf, prev_q,
                                          cc.expand(B, -1), torch.full((B,), kk / N), rtt)
        ret = ret + (0.99 ** j) * tab.gather(1, lv[:, None]).squeeze(1) / 100.0
        buf = (buf - sz[lv] * 8.0 / c - rtt).clamp_min(0) + cl
        prev_q = qh[lv]
    return torch.full((len(roots),), -1e9).scatter_reduce(0, root, ret, reduce="amax")


SESS, LOCK, LOG = {}, threading.Lock(), []


def decide(msg):
    sid, method, k = msg["sid"], msg["method"], int(msg["k"])
    with LOCK:
        if sid not in SESS:
            SESS[sid] = {"env": new_env(), "k": -1}
        s = SESS[sid]
    e = s["env"]
    last = msg.get("last")
    if last and last.get("bytes") and s.get("pending") is not None:       # complete the previous download
        thr = last["bytes"] * 8.0 / max(last["dur_s"], 1e-3) / B_IN_MB   # Mbps, as measured by the player
        lvl_prev, c_used = s["pending"]
        e.thr_hist = np.roll(e.thr_hist, -1); e.thr_hist[-1] = thr
        e.dl_hist = np.roll(e.dl_hist, -1); e.dl_hist[-1] = last["dur_s"]
        if e.cfg.hard_tput == "conformal":                                 # ACI update (as in ABREnv.step)
            e.alpha_t += e.cfg.conf_gamma * (e.cfg.conf_alpha - float(thr * B_IN_MB < c_used))
        e.log.append({"k": k - 1, "level": lvl_prev, "thr": thr, "c_hat": s["c_hat_prev"] / B_IN_MB,
                      "c_low": s["c_low_prev"] / B_IN_MB})
        e.prev_level = lvl_prev
        e.prev_q = float(e._quality_curve(k - 1)[lvl_prev])
    e.k = min(k, e.N - 1); e.buffer = float(msg.get("buffer", 0.0))
    t0 = time.perf_counter()
    e.ctx = e.decision_context()
    lm = int(e.ctx["l_max"])
    if method == "nesy":
        roots = list(range(lm + 1)); sc = plan_scores(e, roots, 3); level = int(roots[int(torch.argmax(sc))])
    elif method == "mpc":
        level = int(RobustMPC(5, "quality").act(e))
    elif method == "bba":
        level = int(BBA().act(e))
    else:
        raise ValueError(method)
    ms = (time.perf_counter() - t0) * 1e3
    s["pending"] = (level, float(e.ctx.get("c_h1", e.ctx["c_low"])))
    s["c_hat_prev"], s["c_low_prev"] = float(e.ctx["c_hat"]), float(e.ctx["c_low"])
    rec = {"sid": sid, "method": method, "k": k, "buffer": e.buffer, "level": level, "l_max": lm, "ms": ms,
           "c_h1": float(e.ctx.get("c_h1", 0)) / B_IN_MB, "c_hat": float(e.ctx["c_hat"]) / B_IN_MB, "alpha_t": float(e.alpha_t)}
    LOG.append(rec)
    return rec


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type"); self.send_header("Access-Control-Allow-Methods", "POST, GET")
        self.end_headers()

    def do_POST(self):
        msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            self._send(200, decide(msg))
        except Exception as ex:  # noqa
            import traceback; traceback.print_exc(); self._send(500, {"error": repr(ex)})

    def do_GET(self):
        if self.path.startswith("/log"):
            sid = self.path.split("sid=")[-1] if "sid=" in self.path else None
            self._send(200, [r for r in LOG if sid is None or r["sid"] == sid])
        else:
            self._send(200, {"ok": True})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8333
    print("ABR server on", port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
