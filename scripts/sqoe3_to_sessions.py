"""Waterloo SQoE-III -> NS-QoE session npz (same keys as make_sessions.py) with real MOS as target.

python scripts/sqoe3_to_sessions.py --root data/sqoe3_raw --out data/sqoe3/sqoe3_sessions.npz
       [--concepts data/sqoe3/sqoe3_concepts.json]   # {content: [[5 concept truths] per 2-s segment]}

Per segment: q = VMAF (from the log), level01 = representation / 10, rebuf = rebuffering_duration (row 0 = start-up
delay), bitrate_kbps. buffer / thr_ratio are not logged; they only enter the hard rules, so neutral values are stored.
Without --concepts all visual truths are 0 (playback rules only).
"""
import _common  # noqa
import argparse, csv, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/sqoe3_raw"); ap.add_argument("--out", default="data/sqoe3/sqoe3_sessions.npz")
ap.add_argument("--concepts", default=None)
a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(a.root, "data.csv"))))
conc = json.load(open(a.concepts)) if a.concepts else {}
contents = sorted({r["content"] for r in rows})
recs = []
for r in rows:
    log = list(csv.DictReader(open(os.path.join(a.root, "streaming_logs", r["streaming_log"]))))
    n = len(log)
    c = np.asarray(conc.get(r["content"], np.zeros((n, 5))), np.float32)[:n]
    recs.append(dict(
        q=np.array([float(x["vmaf"]) for x in log], np.float32),
        level01=np.array([int(x["representation_index"]) / 10.0 for x in log], np.float32),
        rebuf=np.array([float(x["rebuffering_duration"]) for x in log], np.float32),
        buffer=np.full(n, 10.0, np.float32), thr_ratio=np.ones(n, np.float32),
        pos=(np.arange(n) / n).astype(np.float32),
        bitrate_kbps=np.array([float(x["video_bitrate"]) for x in log], np.float32),
        concepts=c, video=contents.index(r["content"]), target=float(r["mos"])))
N = max(len(x["q"]) for x in recs)
pad = lambda k, sh=(): np.stack([np.pad(x[k], [(0, N - len(x[k]))] + [(0, 0)] * len(sh)) for x in recs])
arr = {k: pad(k) for k in ["q", "level01", "rebuf", "buffer", "thr_ratio", "pos", "bitrate_kbps"]}
arr["concepts"] = pad("concepts", (5,))
arr["mask"] = np.stack([np.pad(np.ones(len(x["q"]), np.float32), (0, N - len(x["q"]))) for x in recs])
arr["video"] = np.array([x["video"] for x in recs]); arr["target"] = np.array([x["target"] for x in recs], np.float32)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez_compressed(a.out, **arr)
json.dump(contents, open(a.out.replace(".npz", "_contents.json"), "w"))
print(f"{len(recs)} sessions, {len(contents)} contents, {N} segments -> {a.out}  (concepts: {'yes' if conc else 'no'})")
