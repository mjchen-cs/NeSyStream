"""Real-data assets for the overview figure (Fig. 2): chunk frames, probe crops, rate-quality curves, concepts.

python scripts/make_arch_assets.py --out ../cvpr2027_paper/fig/arch
"""
import _common  # noqa
import argparse, json, os, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", default="data/videos"); ap.add_argument("--out", default="runs/fig/arch")
ap.add_argument("--chunks", default="film_tears_of_steel_seg2:4,ugc_Sports_1080P-0640:6,xiph_blue_sky_1080p25:1,uvg_YachtRide:1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
CON = ["HighMotion", "HighTexture", "SceneCut", "Salient", "Dark"]


def grab(name, level, t, w=None):
    mp4 = os.path.join(a.video_dir, f"{name}_ladder", f"level_{level:02d}.mp4")
    wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", mp4], capture_output=True, text=True, check=True).stdout.strip().split(",")
    W, H = int(wh[0]), int(wh[1])
    vf = [] if w is None else ["-vf", f"scale={w}:-2:flags=lanczos"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", mp4, "-frames:v", "1", *vf, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    if w is None:
        return np.frombuffer(raw, np.uint8).reshape(H, W, 3)
    h = len(raw) // (3 * w)
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


meta = []
for i, spec in enumerate(a.chunks.split(",")):
    name, k = spec.split(":"); k = int(k)
    d = np.load(os.path.join(a.video_dir, name + ".npz"))
    cl = float(d["chunk_len"]); t = k * cl + cl / 2; L = len(d["bitrates_kbps"])
    top = grab(name, L - 1, t, 480)
    h = top.shape[0]; w = int(round(h * 16 / 9)) if top.shape[1] > h * 16 / 9 else top.shape[1]
    plt.imsave(os.path.join(a.out, f"thumb{i}.png"), top[:, (top.shape[1] - w) // 2:(top.shape[1] - w) // 2 + w])
    # probe renditions (levels 0 and 6), same central crop, upscaled to the same size for comparison
    for lv in (0, 6):
        f = grab(name, lv, t, 480)
        hh, ww = f.shape[:2]; ch, cw = hh // 2, ww // 2
        plt.imsave(os.path.join(a.out, f"probe{i}_l{lv}.png"), f[hh // 4:hh // 4 + ch, ww // 4:ww // 4 + cw])
    v, pq, br = d["vmaf"][k], d["pqf_q"][k], d["bitrates_kbps"] / 1000
    fig, ax = plt.subplots(figsize=(1.25, 0.95), dpi=300)
    ax.plot(br, pq, "-", color="#2f6db5", lw=1.2)
    ax.plot(br[[0, 6]], v[[0, 6]], "o", ms=3.2, color="#d0702a", zorder=3)
    ax.plot(br, v, ".", ms=1.6, color="0.55", zorder=2)
    ax.set_xscale("log"); ax.set_ylim(0, 100); ax.set_xticks([]); ax.set_yticks([0, 50, 100])
    ax.tick_params(labelsize=4, length=1.5, pad=1); ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.97, bottom=0.08)
    fig.savefig(os.path.join(a.out, f"curve{i}.pdf")); plt.close(fig)
    c = d["concepts"][k]
    meta.append({"name": name, "chunk": k, "vmaf": v.round(1).tolist(), "pqf": pq.round(1).tolist(),
                 "bitrates_mbps": br.round(3).tolist(), "concepts": {n: round(float(x), 2) for n, x in zip(CON, c)},
                 "probe_vmaf": [round(float(v[0]), 1), round(float(v[6]), 1)]})
    print(name, k, meta[-1]["concepts"], meta[-1]["probe_vmaf"])
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
