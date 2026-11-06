"""E1 pilot, cross-platform (Windows PowerShell / PyCharm terminal / Linux):
PQF variants vs. simple and deep-VQA baselines, grouped 5-fold cross-validation
-> summary table (mean +- std over folds, plus film / short-clip subsets) + cost-accuracy Pareto figure.

python scripts/extract_fragments.py --video_dir data/videos --ref_dir data/refs --jobs 8   (once, ~1.5 GB)
python scripts/extract_rung_features.py --video_dir data/videos --ref_dir data/refs --mode full   (v7)
python scripts/extract_rung_features.py --video_dir data/videos --ref_dir data/refs --mode frag   (v7 ablation)
python scripts/run_e1_pilot.py --video_dir data/videos --out runs/e1_v4
  --folds 5               grouped, source-stratified K-fold CV (films never straddle folds); 0 = old fixed split
  --distill               also train PQF distilled from FineVQ features (needs <video>_finevq.npy)
  --only pqf_mnv3_32      run a subset (comma list of job names; see jobs below)
Finished jobs are skipped when their output exists, so the script can be re-run after an interruption.
"""
import _common  # noqa
import argparse, glob, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(args, log):
    cmd = [sys.executable, os.path.join(HERE, args[0])] + [str(x) for x in args[1:]]
    print(">>", " ".join(cmd), flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode}, see {log})"
    print(f"   {status}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    return p.returncode == 0


def jobs_for(V, R, a, fold):
    E = ["--epochs", a.epochs, "--fold", fold, "--n_folds", max(a.folds, 1)]
    pre = ["--no_pretrained"] if a.no_pretrained else []
    mn = ["--backbone", "mobilenet_v3_small", "--freeze", *pre]          # main model: frozen ImageNet backbone
    pqf = lambda name, *extra: (name, ["train_pqf.py", "--video_dir", V, "--out", os.path.join(R, name + ".pt"),
                                       *E, *extra], os.path.join(R, name + "_metrics.json"))
    frj = lambda name, *extra: (name, ["train_pqf_fr.py", "--video_dir", V, "--out", os.path.join(R, name + ".pt"),
                                       "--fold", fold, "--n_folds", max(a.folds, 1), *extra],
                                os.path.join(R, name + "_metrics.json"))
    base = lambda name, kind, *extra: (name, ["train_feature_baseline.py", "--video_dir", V, "--kind", kind,
                                              "--fold", fold, "--n_folds", max(a.folds, 1), "--out", os.path.join(R, name + ".json"), *extra],
                                       os.path.join(R, name + ".json"))
    fp = ["--frames", "fragpair", "--anchors", "--dropout", 0.1]   # v3 input: fragment pairs + encoder anchors
    ft = ["--frames", "fragtri", "--anchors", "--dropout", 0.1]    # v5 input: source | lowest | probe rung + anchors
    jobs = [
        pqf("pqf_v3", *mn, *fp),                                             # main
        pqf("pqf_v3_z16", *mn, *fp, "--z_dim", 16),                          # descriptor-size sweep
        pqf("pqf_v3_z64", *mn, *fp, "--z_dim", 64),
        pqf("pqf_v3_noanchor", *mn, "--frames", "fragpair", "--dropout", 0.1),   # vision only (no PSNR/SSIM)
        pqf("pqf_v3_srcfrag", *mn, "--frames", "frag", "--anchors", "--dropout", 0.1),  # no lowest-rung pair
        pqf("pqf_v3_thumb", *mn, "--anchors", "--dropout", 0.1),             # old thumbnails + anchors
        pqf("pqf_v2_thumb", *mn),                                            # previous version (reference)
        # v5: fragments of the source, the lowest AND the probe rung -> learned two-rung perceptual FR cue
        pqf("pqf_v5", *mn, *ft, "--z_dim", 16),
        pqf("pqf_v5_noanchor", *mn, "--frames", "fragtri", "--dropout", 0.1, "--z_dim", 16),   # vision only
        pqf("pqf_v5_ft", "--backbone", "mobilenet_v3_small", *pre, *ft, "--z_dim", 16, "--lr", 1e-4),  # fine-tuned
        # v6: two-stage PQF (learned perceptual FR proxy on probe rungs + monotone curve completion)
        frj("pqf_v6", "--feat", "v6", "--label", "PQF-FR (ours)"),
        frj("pqf_v6_nopsnr", "--feat", "v6", "--no_psnr", "--label", "PQF-FR, no PSNR"),               # pure vision
        frj("pqf_v6_probeonly", "--feat", "v6", "--fr_train_rungs", "probes", "--label", "PQF-FR, probe-rung training"),
        frj("pqf_v6_1probe", "--feat", "v6", "--probes", "0", "--label", "PQF-FR, 1 probe"),
        frj("pqf_v6_3probe", "--feat", "v6", "--probes", "0,4,8", "--label", "PQF-FR, 3 probes"),
        frj("base_psnr_probes", "--feat", "v6", "--no_fr", "--label", "PSNR at 2 probes"),
        frj("base_oracle_probes", "--feat", "v6", "--oracle_fr", "--label", "true VMAF at 2 probes"),
        # v7: whole keyframes at native resolution + LPIPS-style (compare-then-pool) distortion features
        frj("pqf_v7", "--label", "PQF-FR (ours)"),
        frj("pqf_v7_nopsnr", "--no_psnr", "--label", "PQF-FR, pure vision"),
        frj("pqf_v7_probeonly", "--fr_train_rungs", "probes", "--label", "PQF-FR, probe-rung training"),
        frj("pqf_v7_3probe", "--probes", "0,4,8", "--label", "PQF-FR, 3 probes"),
        frj("pqf_v7_frag", "--feat", "frag", "--label", "PQF-FR, fragments"),
        frj("pqf_v7_frag_nopsnr", "--feat", "frag", "--no_psnr", "--label", "PQF-FR, fragments, pure vision"),
        frj("base_psnr_3probes", "--no_fr", "--probes", "0,4,8", "--label", "PSNR at 3 probes"),
        base("base_bitrate", "bitrate"),
        base("base_handcrafted", "handcrafted"),
        base("base_anchors", "anchors"),
        base("base_vmaf_lowest", "vmaf_lowest"),
        base("base_vmaf_two", "vmaf_two"),
    ]
    for F in ("finevq", "dover", "fastvqa"):  # deep-VQA baselines, only if their features were imported
        if glob.glob(os.path.join(V, f"*_{F}.npy")):
            jobs.append(base(f"base_{F}_feat", "features", "--feat_name", F))
        if glob.glob(os.path.join(V, f"*_{F}_scores.npy")):
            jobs.append(base(f"base_{F}_all", "per_rendition", "--feat_name", F))
    if a.distill:
        jobs.append(pqf("pqf_distill_finevq", *mn, *fp, "--teacher", "finevq"))
    return jobs


def flat(r):
    """uniform view of a train_pqf / train_feature_baseline result."""
    if "test" in r:
        t = r["test"]; s = r.get("test_subsets", {})
        return {"mae": t["mae"], "srocc": t["srocc"], "srocc_c": t["srocc_content"], "film": s.get("film", np.nan),
                "short": s.get("short", np.nan), "train": r.get("train_mae", np.nan), "bytes": t["bytes_est"],
                "ms": r["cost"]["ms_per_chunk"]}
    s = r.get("subsets", {})
    return {"mae": r["mae"], "srocc": r["srocc"], "srocc_c": r["srocc_content"], "film": s.get("film", np.nan),
            "short": s.get("short", np.nan), "train": np.nan, "bytes": r["cost"]["bytes_per_chunk"],
            "ms": r["cost"]["ms_per_chunk"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="data/videos"); ap.add_argument("--out", default="runs/e1_v4")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--distill", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--no_pretrained", action="store_true", help="offline machines: MobileNetV3 without ImageNet weights")
    a = ap.parse_args()
    V, R = a.video_dir, a.out
    folds = list(range(a.folds)) if a.folds > 0 else [-1]
    only = set(filter(None, a.only.split(",")))
    names = []
    for k in folds:
        Rk = os.path.join(R, f"fold{k}" if k >= 0 else "fixed"); os.makedirs(Rk, exist_ok=True)
        for name, args, done_file in jobs_for(V, Rk, a, k):
            if name not in names:
                names.append(name)
            if only and name not in only:
                continue
            if os.path.exists(done_file):
                print(f"-- skip {name} fold {k} (done)"); continue
            print(f"[fold {k}] {name}")
            run(args, os.path.join(Rk, name + ".log"))

    # ---- aggregate over folds
    cols = ["mae", "srocc", "srocc_c", "film", "short", "train", "bytes", "ms"]
    head = f"{'method':22s} {'folds':>5s} " + " ".join(f"{c:>13s}" for c in cols)
    lines, agg_files = [head], []
    os.makedirs(os.path.join(R, "agg"), exist_ok=True)
    for name in names:
        rs, raw = [], None
        for k in folds:
            d = os.path.join(R, f"fold{k}" if k >= 0 else "fixed")
            p = os.path.join(d, name + ("_metrics.json" if name.startswith("pqf") or name.endswith("probes") else ".json"))
            if os.path.exists(p):
                raw = json.load(open(p)); rs.append(flat(raw))
        if not rs:
            continue
        M = {c: np.array([r[c] for r in rs], float) for c in cols}
        fmt = lambda x: f"{np.nanmean(x):6.2f}+-{np.nanstd(x):4.2f}" if np.isfinite(x).any() else f"{'-':>11s}"
        lines.append(f"{name:22s} {len(rs):5d} " + " ".join(f"{fmt(M[c]):>13s}" for c in cols))
        # fold-averaged json in the format plot_e1_pareto.py reads
        mean = {c: float(np.nanmean(M[c])) if np.isfinite(M[c]).any() else float("nan") for c in cols}
        if "test" in raw:
            out = {**raw, "test": {**raw["test"], "mae": mean["mae"], "srocc": mean["srocc"],
                                   "srocc_content": mean["srocc_c"], "bytes_est": mean["bytes"]},
                   "cost": {**raw["cost"], "ms_per_chunk": mean["ms"]}}
        else:
            out = {**raw, "mae": mean["mae"], "srocc": mean["srocc"], "srocc_content": mean["srocc_c"],
                   "cost": {**raw["cost"], "ms_per_chunk": mean["ms"]}}
        out["n_folds"] = len(rs); out["fold_std_mae"] = float(np.nanstd(M["mae"]))
        f = os.path.join(R, "agg", name + ".json"); json.dump(out, open(f, "w"), indent=1); agg_files.append(f)
    lines.append("mae/srocc on all test chunks; film/short = MAE on film segments / short clips; "
                 "train = MAE on the training split (generalisation gap); mean+-std over folds")
    print("\n".join(lines))
    open(os.path.join(R, "summary.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if agg_files:
        run(["plot_e1_pareto.py", "--results", *[f for f in agg_files if not any(t in os.path.basename(f) for t in ("_v2_", "_v3", "_v5", "_v6", "oracle"))],
             "--out", os.path.join(R, "e1_pareto.pdf")], os.path.join(R, "plot.log"))
    print("summary:", os.path.join(R, "summary.txt"), " figure:", os.path.join(R, "e1_pareto.pdf"))


if __name__ == "__main__":
    main()
