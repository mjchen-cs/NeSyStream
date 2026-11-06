import json, os, subprocess, numpy as np, sys
sys.path.insert(0, "scripts"); import _common  # noqa
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
VD = "data/videos"; OUT = "../cvpr2027_paper/fig/teaser3"; os.makedirs(OUT, exist_ok=True)
def grab(name, level, t, w=640):
    mp4 = f"{VD}/{name}_ladder/level_{level:02d}.mp4"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", mp4, "-frames:v", "1", "-vf",
                          f"scale={w}:-2:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, w, 3)
def crop169(img, f=1.0):
    h, w = img.shape[:2]; cw = int(w * f); ch = int(cw * 9 / 16)
    if ch > h: ch = h; cw = int(ch * 16 / 9)
    y, x = (h - ch) // 2, (w - cw) // 2; return img[y:y + ch, x:x + cw]
grid = ["film_tears_of_steel_seg2:4", "xiph_pedestrian_area_1080p25:2", "xiph_old_town_cross_1080p50:2", "film_tears_of_steel_seg1:3",
        "xiph_in_to_tree_1080p50:2", "uvg_ShakeNDry:0", "uvg_YachtRide:0", "ugc_Sports_1080P-0640:6"]
meta = {"grid": []}
for i, s in enumerate(grid):
    n, k = s.split(":"); k = int(k); d = np.load(f"{VD}/{n}.npz"); cl = float(d["chunk_len"])
    img = grab(n, 5, k * cl + cl / 2)
    plt.imsave(f"{OUT}/g{i}.jpg", crop169(img), pil_kwargs={"quality": 92})
    meta["grid"].append({"name": n, "chunk": k, "vmaf": round(float(d["vmaf"][k, 5]), 1)})
    if i in (0, 7):   # zoom crops (native resolution of level 5)
        full = grab(n, 5, k * cl + cl / 2, w=960); h, w = full.shape[:2]
        cy, cx = (int(h * 0.4), int(w * 0.63)) if i == 0 else (int(h * 0.5), int(w * 0.45))
        ch = h // 2 if i == 0 else h // 3; cw = int(ch * 4 / 3)
        plt.imsave(f"{OUT}/z{i}.png", full[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2])
S = json.load(open("runs/fig/teaser_sessions_v3_sessions.json"))
ours = next(r for r in S["Ours-plan3-all"] if r["video"] == "film_tears_of_steel_seg2" and r["trace"] == "norway_ferry_17")
meta["strip"] = []
for j, c in enumerate([1, 5, 8, 21, 24, 27]):
    e = ours["log"][c]; n, k = e["src"][0], int(e["src"][1]); d = np.load(f"{VD}/{n}.npz"); cl = float(d["chunk_len"])
    img = grab(n, int(e["level"]), k * cl + cl / 2, w=480)
    plt.imsave(f"{OUT}/s{j}.jpg", crop169(img), pil_kwargs={"quality": 92})
    meta["strip"].append({"chunk": c, "src": [n, k], "level": int(e["level"]), "vmaf": round(float(e["vmaf"]), 1)})
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1); print(json.dumps(meta))
