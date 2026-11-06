"""Unpack the trace zips into data/traces/{train,test/<set>} and drop test traces that are
byte-identical to any training trace (content leakage).  Standard library only.

python scripts/prepare_traces.py --zip_dir ABR-data/traces --out data/traces
"""
import argparse, hashlib, io, json, os, zipfile

TRAIN = ["puffer.zip"]
TEST = {"hsdpa": "hsdpa.zip", "fcc": "fcc.zip", "oboe": "oboe.zip", "lumos4g": "lumos4g.zip", "lumos5g": "5g.zip"}

ap = argparse.ArgumentParser()
ap.add_argument("--zip_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--also_check", default="train.zip", help="extra zips whose traces must not appear in test (comma list)")
a = ap.parse_args()


def members(zpath):
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith("/"):
                continue
            data = z.read(n)
            if not data.strip():
                continue
            yield os.path.basename(n), data


def valid(data):
    """At least two numeric 'time throughput' rows."""
    rows = 0
    for line in io.StringIO(data.decode(errors="ignore")):
        p = line.split()
        if len(p) >= 2:
            try:
                float(p[0]); float(p[1]); rows += 1
            except ValueError:
                return False
    return rows >= 2


train_hashes, report = set(), {"train": 0, "test": {}, "removed_duplicates": [], "removed_invalid": []}
os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
for zn in TRAIN:
    for name, data in members(os.path.join(a.zip_dir, zn)):
        if not valid(data):
            report["removed_invalid"].append(f"{zn}:{name}"); continue
        train_hashes.add(hashlib.md5(data).hexdigest())
        open(os.path.join(a.out, "train", name), "wb").write(data); report["train"] += 1
for zn in filter(None, a.also_check.split(",")):
    p = os.path.join(a.zip_dir, zn)
    if os.path.exists(p):
        for _, data in members(p):
            train_hashes.add(hashlib.md5(data).hexdigest())
for tag, zn in TEST.items():
    d = os.path.join(a.out, "test", tag); os.makedirs(d, exist_ok=True); n = 0
    for name, data in members(os.path.join(a.zip_dir, zn)):
        if not valid(data):
            report["removed_invalid"].append(f"{zn}:{name}"); continue
        if hashlib.md5(data).hexdigest() in train_hashes:
            report["removed_duplicates"].append(f"{zn}:{name}"); continue
        open(os.path.join(d, name), "wb").write(data); n += 1
    report["test"][tag] = n
json.dump(report, open(os.path.join(a.out, "prepare_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in report.items() if k != "removed_invalid"}, indent=1, ensure_ascii=False))
print("invalid/empty files skipped:", len(report["removed_invalid"]))
