"""Chunk-level torch dataset for PQF and split helpers."""
from __future__ import annotations

import glob
import re
import json
import os
import numpy as np
try:  # torch is only needed for ChunkDataset; split helpers work without it
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch, Dataset = None, object


def list_videos(video_dir):
    """per-video npz files (side files such as <video>_frag.npz are excluded)."""
    return sorted(p for p in glob.glob(os.path.join(video_dir, "*.npz")) if not is_side_file(p))


def is_side_file(p):
    return bool(re.search(r"_(frag|rungfeat\w*)\.npz$", p))


def split_by_video(paths, split_json=None, val=0.15, test=0.25, seed=0, groups=None):
    """Split by SOURCE content (never by chunk) to avoid content leakage.
    split_json: optional {"train": [names], "val": [...], "test": [...]}.
    groups: optional {"group": [names that show the same content]} (e.g. two renders of Big Buck Bunny);
            members of a group always land in the same split. Default: data/videos/groups.json if present."""
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if split_json and os.path.exists(split_json):
        s = json.load(open(split_json))
        pick = lambda k: [p for p, n in zip(paths, names) if n in set(s[k])]
        return pick("train"), pick("val"), pick("test")
    if len(paths) < 3:
        print("[warn] fewer than 3 videos: using all of them for train/val/test (testing only)")
        return list(paths), list(paths), list(paths)
    if groups is None and paths:
        gj = os.path.join(os.path.dirname(paths[0]), "groups.json")
        groups = json.load(open(gj)) if os.path.exists(gj) else {}
    owner = {n: g for g, mem in (groups or {}).items() for n in mem}
    units = {}
    for p, n in zip(paths, names):
        units.setdefault(owner.get(n, n), []).append(p)
    def n_chunks(p):
        try:
            with np.load(p) as d:
                return int(d["sizes"].shape[0])
        except Exception:
            return 1
    size = {k: sum(n_chunks(p) for p in v) for k, v in units.items()}
    assign = assign_units(size, {k: os.path.splitext(os.path.basename(v[0]))[0].split("_")[0] for k, v in units.items()},
                          {"train": 1 - val - test, "val": val, "test": test}, seed)
    out = {sp: [] for sp in ("train", "val", "test")}
    for k, sp in assign.items():
        out[sp] += units[k]
    return sorted(out["train"]), sorted(out["val"]), sorted(out["test"])


def assign_units(size, source, target, seed=0, big_frac=0.04):
    """Assign content units (a clip, or a group of segments of one film) to splits.
    * small units are stratified BY SOURCE (UGC, Xiph, UVG, ...): every source is split ~target-proportionally,
      so train/val/test all contain every kind of content;
    * large units (whole films) are few: every placement is enumerated and the one with the best score is kept:
      score = 3 * |fraction error| + 2 * (share of the largest unit in test) + 1 * (same in val)."""
    import itertools
    order = list(target)
    total = sum(size.values())
    big = sorted([k for k in size if size[k] > big_frac * total], key=lambda k: -size[k])[:9]
    rng = np.random.default_rng(seed)
    got = {sp: 0 for sp in order}; mx = {sp: 0 for sp in order}; base = {}
    for src in sorted({source[k] for k in size if k not in big}):
        us = [k for k in size if k not in big and source[k] == src]
        us = [us[i] for i in rng.permutation(len(us))]
        us.sort(key=lambda k: -size[k])
        g = {sp: 0 for sp in order}; tsrc = sum(size[k] for k in us)
        for k in us:
            sp = max(order, key=lambda q: target[q] * tsrc - g[q])
            g[sp] += size[k]; base[k] = sp; got[sp] += size[k]; mx[sp] = max(mx[sp], size[k])
    best = None
    for combo in itertools.product(order, repeat=len(big)):
        g = dict(got); m = dict(mx)
        for k, sp in zip(big, combo):
            g[sp] += size[k]; m[sp] = max(m[sp], size[k])
        if any(g[sp] < (0.66 * target[sp]) * total for sp in order):   # every split keeps a usable size
            continue
        frac_err = sum(abs(g[sp] / total - target[sp]) for sp in order)
        score = 3 * frac_err + 2 * m["test"] / g["test"] + 1 * m["val"] / g["val"]
        if best is None or score < best[0]:
            best = (score, dict(zip(big, combo)))
    base.update(best[1] if best else {})
    return base


def _units(paths):
    """content units: a stand-alone clip, or all segments of one film / production (groups.json)."""
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    gj = os.path.join(os.path.dirname(paths[0]), "groups.json") if paths else ""
    groups = json.load(open(gj)) if gj and os.path.exists(gj) else {}
    owner = {n: g for g, mem in groups.items() for n in mem}
    units = {}
    for p, n in zip(paths, names):
        units.setdefault(owner.get(n, n), []).append(p)
    return units


def source_of(path):
    """coarse source label used for stratification and subset reporting ('film' / 'ugc' / 'xiph' / ...)."""
    n = os.path.splitext(os.path.basename(path))[0]
    return n.split("_")[0]


def kfold_assign(paths, n_folds=5, seed=0):
    """Grouped, source-stratified K-fold assignment {path: fold}.
    Large units (whole films, >4 % of the data) are spread over folds first (largest unit -> emptiest fold);
    the remaining clips are then dealt per source so that every fold sees every kind of content."""
    units = _units(paths)
    def n_chunks(p):
        with np.load(p) as d:
            return int(d["sizes"].shape[0])
    size = {k: sum(n_chunks(p) for p in v) for k, v in units.items()}
    total = sum(size.values())
    rng = np.random.default_rng(seed)
    load = np.zeros(n_folds); fold = {}
    big = sorted([k for k in size if size[k] > 0.04 * total], key=lambda k: -size[k])
    for k in big:
        f = int(np.argmin(load + 1e-3 * rng.random(n_folds))); fold[k] = f; load[f] += size[k]
    small = [k for k in size if k not in fold]
    for src in sorted({source_of(units[k][0]) for k in small}):
        us = [k for k in small if source_of(units[k][0]) == src]
        us = [us[i] for i in rng.permutation(len(us))]
        us.sort(key=lambda k: -size[k])
        g = np.zeros(n_folds)
        for k in us:   # balance this source across folds, break ties by total fold size
            f = int(np.lexsort((load, g))[0]); fold[k] = f; g[f] += size[k]; load[f] += size[k]
    return {p: fold[k] for k, v in units.items() for p in v}


def get_split(video_dir, split_json=None, fold=-1, n_folds=5, seed=0):
    """fold < 0: the fixed split (split.json). fold = k: test = fold k, val = fold k+1, train = the rest.
    The fold assignment is cached in <video_dir>/folds<n>.json so every script uses the same folds."""
    paths = list_videos(video_dir)
    if fold < 0:
        return split_by_video(paths, split_json or os.path.join(video_dir, "split.json"))
    fj = os.path.join(video_dir, f"folds{n_folds}.json")
    if os.path.exists(fj):
        a = json.load(open(fj))
    else:
        a = {os.path.splitext(os.path.basename(p))[0]: f for p, f in kfold_assign(paths, n_folds, seed).items()}
        json.dump(a, open(fj, "w"), indent=0)
    nm = lambda p: os.path.splitext(os.path.basename(p))[0]
    te = [p for p in paths if a.get(nm(p)) == fold]
    va = [p for p in paths if a.get(nm(p)) == (fold + 1) % n_folds]
    tr = [p for p in paths if nm(p) in a and a[nm(p)] not in (fold, (fold + 1) % n_folds)]
    return tr, va, te


def chunk_sources(paths):
    """per-chunk source label, in the same order as ChunkDataset concatenates the chunks."""
    out = []
    for p in paths:
        with np.load(p) as d:
            out += [source_of(p)] * int(d["vmaf"].shape[0])
    return np.array(out)


def subset_mae(pred, target, src):
    """MAE on all chunks and on the film / short-clip subsets (paper reports all three)."""
    e = np.abs(pred - target).mean(1)
    film = src == "film"
    return {"all": float(e.mean()), "film": float(e[film].mean()) if film.any() else float("nan"),
            "short": float(e[~film].mean()) if (~film).any() else float("nan")}


class ChunkDataset(Dataset):
    def __init__(self, npz_paths, cmp_stats=None, teacher: str = "", frames: str = "thumb", anchors: bool = False):
        """teacher: name of per-video teacher features '<video>_<teacher>.npy' (e.g. 'finevq'), optional.
        frames : 'thumb'    - 128x128 keyframes of the lowest rung (original PQF input)
                 'frag'     - native-resolution source fragments (<video>_frag.npz, scripts/extract_fragments.py)
                 'fragpair' - source fragments and the lowest-rung fragments, stacked as 6 channels
                 'fragtri'  - source | lowest rung | probe rung fragments, 9 channels
        anchors: append the encoder-side PSNR/SSIM anchors (<video>_frag.npz) to the bitstream statistics."""
        fr, cmp, q, c, t = [], [], [], [], []
        for p in npz_paths:
            d = np.load(p)
            side = np.load(p[:-4] + "_frag.npz") if (frames != "thumb" or anchors) else None
            if frames == "thumb":
                fr.append(d["frames"])
            elif frames == "frag":
                fr.append(side["frag_src"])
            elif frames == "fragpair":
                fr.append(np.concatenate([side["frag_src"], side["frag_low"]], axis=2))
            else:  # fragtri
                fr.append(np.concatenate([side["frag_src"], side["frag_low"], side["frag_mid"]], axis=2))
            x = d["cmp_feats"]
            if anchors:
                x = np.concatenate([x, side["anchors"]], 1)
            cmp.append(x); q.append(d["vmaf"])
            c.append(d["concepts"] if "concepts" in d else np.zeros((len(d["vmaf"]), 5), np.float32))
            if teacher:
                tf = np.load(p[:-4] + f"_{teacher}.npy").astype(np.float32)
                t.append(tf / (np.linalg.norm(tf, axis=1, keepdims=True) + 1e-6))
        self.t = np.concatenate(t) if teacher else None
        self.frames = np.concatenate(fr)
        cmp = np.concatenate(cmp).astype(np.float32)
        self.cmp_stats = cmp_stats or (cmp.mean(0), cmp.std(0) + 1e-6)
        self.cmp = (cmp - self.cmp_stats[0]) / self.cmp_stats[1]
        self.q = np.concatenate(q).astype(np.float32)
        self.c = np.concatenate(c).astype(np.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        t = torch.from_numpy(self.t[i]) if self.t is not None else torch.zeros(0)
        return (torch.from_numpy(self.frames[i]), torch.from_numpy(self.cmp[i]), torch.from_numpy(self.q[i]),
                torch.from_numpy(self.c[i]), t)


def _rank(x):
    """Average ranks with ties (as in scipy.stats.rankdata)."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=np.float64)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / cnt[inv]


def srocc(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def plcc(a, b):
    return float(np.corrcoef(a, b)[0, 1])
