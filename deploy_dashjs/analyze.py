"""Metrics of the dash.js deployment sessions (runs_dep/*.json).
Per session: levels / bytes from the segment server log, stalls from the video element's waiting/playing events
(start-up excluded), true VMAF of the downloaded renditions, ITU-T P.1203 (mode 0), decision latency, H1 violations."""
import json, glob, sys, collections
import numpy as np
P = json.load(open("src/playlist.json")); VM = np.array(P["vmaf"]); BR = np.array(P["bitrates_kbps"]); CL = P["chunk_len"]
HEIGHTS = P["heights"]
RES = [f"{int(h * 2.4) // 2 * 2}x{h}" for h in HEIGHTS]     # 2.39:1 film


def p1203(levels, stalls):
    from itu_p1203 import P1203Standalone
    import logging; logging.disable(logging.WARNING)
    segs = [{"bitrate": float(BR[l]), "codec": "h264", "duration": CL, "fps": 24.0, "resolution": RES[l], "start": i * CL}
            for i, l in enumerate(levels)]
    inp = {"I11": {"segments": [], "streamId": 1}, "I13": {"segments": segs, "streamId": 1},
           "I23": {"stalling": [[float(p), float(d)] for p, d in stalls], "streamId": 1},
           "IGen": {"device": "pc", "displaySize": "1920x1080", "viewingDistance": "150cm"}}
    return float(P1203Standalone(inp).calculate_complete()["O46"])


def session(fn):
    d = json.load(open(fn)); seg = {s["k"]: s for s in d["seg"]}; ev = d["log"]["events"]
    if len(seg) < 30 or not d["log"]["done"]:
        return None
    lv = [seg[k]["level"] for k in range(30)]
    first_play = next((t for e, t, _ in ev if e == "playing"), None)
    stalls, startup, w = [], first_play, None
    for e, t, ct in ev:
        if first_play is None or t <= first_play:
            continue
        if e == "waiting" and w is None:
            w = (t, ct)
        elif e == "playing" and w is not None:
            stalls.append((w[1], t - w[0])); w = None
    stall_s = sum(x for _, x in stalls)
    dec = d["log"]["decisions"]
    q = VM[np.arange(30), lv]
    st = [(0.0, startup)] + stalls if startup else stalls
    return {"method": d["method"], "set": d["trace"].split("/")[-2], "trace": d["trace"].split("/")[-1],
            "vmaf": float(q.mean()), "switch": float(np.mean(np.abs(np.diff(q)))), "mb": sum(seg[k]["bytes"] for k in range(30)) / 1e6,
            "stall_s": stall_s, "n_stalls": len(stalls), "stall_ratio": stall_s / (stall_s + 60.0), "startup_s": startup,
            "p1203": p1203(lv, st), "ms": float(np.mean([x["ms"] for x in dec])) if dec else float("nan"),
            "viol": float(np.mean([x["level"] > x["l_max"] for x in dec])) if dec else float("nan"),
            "dropped": d["quality"].get("dropped")}


if __name__ == "__main__":
    rows = [r for r in (session(f) for f in sorted(glob.glob("runs_dep/*.json")) if not f.endswith("seglog.json")) if r]
    json.dump(rows, open("dep_rows.json", "w"), indent=0)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["set"], r["method"])].append(r); by[("all", r["method"])].append(r)
    keys = ["p1203", "vmaf", "stall_ratio", "n_stalls", "startup_s", "switch", "mb", "viol", "ms"]
    for s in ["hsdpa", "fcc", "lumos4g", "lumos5g", "all"]:
        for m in ["nesy", "mpc", "bba", "dashjs"]:
            R = by.get((s, m), [])
            if R:
                print(f"{s:8s} {m:7s} n={len(R):2d} " + " ".join(f"{k} {np.nanmean([r[k] for r in R]):.3f}" for k in keys))
    # paired differences (same trace) nesy - other
    idx = {(r["set"], r["trace"], r["method"]): r for r in rows}
    for o in ["mpc", "bba", "dashjs"]:
        d = [idx[(s, t, "nesy")]["p1203"] - idx[(s, t, o)]["p1203"] for (s, t, m) in idx if m == "nesy" and (s, t, o) in idx]
        ds = [idx[(s, t, "nesy")]["stall_ratio"] - idx[(s, t, o)]["stall_ratio"] for (s, t, m) in idx if m == "nesy" and (s, t, o) in idx]
        if d:
            rng = np.random.default_rng(0); bs = [np.mean(rng.choice(d, len(d))) for _ in range(4000)]
            print(f"nesy - {o:7s} P.1203 {np.mean(d):+.3f} [{np.quantile(bs, .025):+.3f}, {np.quantile(bs, .975):+.3f}] (n={len(d)}), "
                  f"stall ratio {100 * np.mean(ds):+.2f} pp, wins {np.mean(np.array(d) > 0):.0%}")
