"""Cost-accuracy Pareto figure for E1 (paper Fig.: PQF vs. deep VQA baselines).

python scripts/plot_e1_pareto.py --results runs/e1/*.json --out ../cvpr2027_paper/fig/e1_pareto.pdf
Each json is the output of train_pqf.py (*_metrics.json) or train_feature_baseline.py.
x = server cost per chunk (ms, log scale), y = MAE to VMAF; the Pareto front is drawn as a step line.
"""
import argparse, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--results", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
GROUP = {"ours": ("Ours (PQF)", "#2a78d6"), "deep": ("Deep VQA baselines", "#eb6834"), "simple": ("Simple baselines", "#898781")}
pts = []
for p in a.results:
    r = json.load(open(p))
    if "test" in r:  # train_pqf metrics
        cfg = r.get("config", {})
        name = "PQF" + ("" if cfg.get("backbone", "scratch") == "scratch" else f" ({cfg['backbone'].replace('mobilenet_v3_small', 'MNv3')}{'' if cfg.get('freeze') else ', ft'})") \
               + ("" if cfg.get("inputs", "both") == "both" else f", {cfg['inputs']} only") \
               + (f", distilled" if cfg.get("teacher") else "") + f", {int(round(r['test']['bytes_est']))} B"
        if cfg.get("label"):
            name = f"{cfg['label']}, {int(round(r['test']['bytes_est']))} B"
        grp = "simple" if (cfg.get("no_fr") or cfg.get("oracle_fr")) else "ours"
        pts.append((name, r["cost"]["ms_per_chunk"], r["test"]["mae"], grp))
    else:
        g = "simple" if r["kind"] in ("bitrate", "handcrafted", "anchors", "vmaf_lowest", "vmaf_two") else "deep"
        pts.append((r["name"], r["cost"]["ms_per_chunk"], r["mae"], g))
pts = [p for p in pts if p[1] is not None and not math.isnan(p[1]) and not math.isnan(p[2])]
pts = [(n, max(c, 1e-2), m, g) for n, c, m, g in pts]          # zero-cost prior drawn at the left edge
fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#898781"); ax.spines[s].set_linewidth(0.6)
ax.grid(True, which="major", color="#e4e3df", linewidth=0.5); ax.set_axisbelow(True)
front, best = [], float("inf")
for n, c, m, g in sorted(pts, key=lambda t: t[1]):
    if m < best:
        best = m; front.append((c, m))
if len(front) > 1:
    xs, ys = zip(*front)
    ax.step(xs, ys, where="post", color="#b0afa8", linewidth=1, zorder=1)
seen = set()
for n, c, m, g in pts:
    lab, col = GROUP[g]
    ax.scatter([c], [m], s=28, color=col, edgecolor="white", linewidth=1.2, zorder=3, label=None if g in seen else lab)
    seen.add(g)
    ax.annotate(n, (c, m), xytext=(4, 3), textcoords="offset points", fontsize=5.5, color="#3d3c37")
ax.set_xscale("log")
ax.set_xlabel("Server cost per chunk (ms, log scale)", fontsize=7, color="#3d3c37")
ax.set_ylabel("MAE to VMAF (lower is better)", fontsize=7, color="#3d3c37")
ax.tick_params(labelsize=6, colors="#5f5e59", width=0.6)
ax.legend(fontsize=6, frameon=False, loc="upper right")
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
fig.savefig(a.out); print("saved", a.out)
