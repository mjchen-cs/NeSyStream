"""Write the server-side PQF outputs into every video npz for the streaming experiments (E2/E3):
  pqf_q        (N, L)  predicted rate-quality curve      main PQF: true VMAF on 2 probe rungs + monotone completion
  pqf_z        (N, 2)  shipped descriptor                the 2 probe qualities (0.5-VMAF steps, 1 byte each)
  pqf_q_cheap, pqf_z_cheap                               cheap PQF (v7: learned perceptual proxy at the probes)
  pqf_concepts (N, C)  visual concept predicates         detector-grounded soft labels (TI, SI, histogram cut,
                                                          YuNet faces, luma) = 'concepts' from build_dataset.py
All curves are OUT-OF-FOLD: every video is predicted by the fold model that never saw it (grouped 5-fold CV).

python scripts/cache_quality.py --video_dir data/videos --work runs/quality
"""
import _common  # noqa
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"": ["--oracle_fr", "--feat", "full", "--label", "PQF (2 VMAF probes)"],
            "cheap": ["--feat", "full", "--label", "PQF-FR (cheap)"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", default="data/videos"); ap.add_argument("--work", default="runs/quality")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    preds = {tag: {} for tag in VARIANTS}
    for tag, extra in VARIANTS.items():
        for k in range(a.folds):
            out = os.path.join(a.work, f"{tag or 'main'}_fold{k}.pt")
            pf = os.path.splitext(out)[0] + "_preds.npz"
            if not os.path.exists(pf):
                cmd = [sys.executable, os.path.join(HERE, "train_pqf_fr.py"), "--video_dir", a.video_dir, "--out", out,
                       "--fold", str(k), "--n_folds", str(a.folds), "--save_preds", *extra]
                print(">>", " ".join(cmd), flush=True)
                subprocess.run(cmd, check=True)
            with np.load(pf) as d:
                for key in d.files:
                    preds[tag][key] = d[key]
            m = json.load(open(os.path.splitext(out)[0] + "_metrics.json"))
            print(f"   [{tag or 'main'} fold {k}] test MAE {m['test']['mae']:.2f}")
    folds = json.load(open(os.path.join(a.video_dir, f"folds{a.folds}.json")))
    for n in sorted(folds):
        p = os.path.join(a.video_dir, n + ".npz")
        d = dict(np.load(p, allow_pickle=False))
        for tag in VARIANTS:
            suf = f"_{tag}" if tag else ""
            d["pqf_q" + suf] = preds[tag][n + "__q"]; d["pqf_z" + suf] = preds[tag][n + "__z"]
        d["pqf_concepts"] = d["concepts"].astype(np.float32)
        np.savez_compressed(p, **d)
        e = {tag or "main": float(np.abs(d["pqf_q" + (f"_{tag}" if tag else "")] - d["vmaf"]).mean()) for tag in VARIANTS}
        print(f"{n:44s} MAE " + "  ".join(f"{t} {v:5.2f}" for t, v in e.items()), flush=True)
    print("done: pqf_q / pqf_z / pqf_q_cheap / pqf_z_cheap / pqf_concepts written to", len(folds), "videos")


if __name__ == "__main__":
    main()
