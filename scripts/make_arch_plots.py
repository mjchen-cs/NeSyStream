"""Plots for Fig. 2, drawn at their final size (fonts ~5 pt).
Real data: rule weights (SQoE-III bootstrap), chunk rate-quality curve (meta.json from make_arch_assets.py).
Illustrative: denoising trajectories, critic values, throughput sparkline."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out, boot = sys.argv[1], json.load(open(sys.argv[2])); meta = json.load(open(f"{out}/meta.json"))
B, O, G, R = "#2f6db5", "#d0702a", "#3a9a5b", "#c8452f"
cm = 1 / 2.54
plt.rcParams.update({"font.size": 5, "axes.linewidth": 0.45, "xtick.major.width": 0.4, "ytick.major.width": 0.4,
                     "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "font.family": "DejaVu Sans"})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=1.6, pad=1)
# rate-quality curve of chunk k (real)
m = meta[0]; br = np.array(m["bitrates_mbps"]); v = np.array(m["vmaf"]); pq = np.array(m["pqf"])
fig, ax = plt.subplots(figsize=(2.35 * cm, 1.75 * cm), dpi=300)
ax.plot(br, v, ".", ms=2.0, color="0.6"); ax.plot(br, pq, "-", lw=1.0, color=B); ax.plot(br[[0, 6]], v[[0, 6]], "o", ms=2.8, color=O)
ax.set_xscale("log"); ax.minorticks_off(); ax.set_ylim(55, 100); ax.set_yticks([60, 80, 100])
ax.set_xticks([0.25, 1, 4, 16]); ax.set_xticklabels(["0.25", "1", "4", "16"]); ax.set_xlabel("Mbps", labelpad=0.5)
ax.set_ylabel("VMAF", labelpad=0.5); clean(ax)
fig.subplots_adjust(left=0.3, right=0.97, top=0.96, bottom=0.27); fig.savefig(f"{out}/curve0.pdf"); plt.close(fig)
# learned rule weights (real, SQoE-III bootstrap)
rules = [("rebuffer_recency", r"Rebuffer$\wedge$NearEnd"), ("rebuffer", "Rebuffer"), ("low_rate", "LowRate"),
         ("salient_drop", r"Salient$\wedge$QualityDrop"), ("switch_mid_shot", r"Switch$\wedge\neg$SceneCut"),
         ("switch_at_cut", r"Switch$\wedge$SceneCut")]
fig, ax = plt.subplots(figsize=(4.4 * cm, 2.15 * cm), dpi=300)
for i, (k, lab) in enumerate(rules):
    md, lo, hi = (max(boot[k][x], 1e-3) for x in ("median", "q05", "q95"))
    ax.barh(i, md - 1e-3, left=1e-3, color=B if i < 3 else O, height=0.62); ax.plot([lo, hi], [i, i], color="k", lw=0.5)
ax.set_xscale("log"); ax.set_xlim(1e-3, 300); ax.minorticks_off(); ax.set_yticks(range(len(rules)))
ax.set_yticklabels([l for _, l in rules]); ax.invert_yaxis(); ax.set_xticks([1e-2, 1, 100]); ax.set_xticklabels(["0.01", "1", "100"])
ax.set_xlabel(r"learned weight $w_\rho$", labelpad=0.5); clean(ax)
fig.subplots_adjust(left=0.47, right=0.97, top=0.98, bottom=0.23); fig.savefig(f"{out}/rules.pdf"); plt.close(fig)
# denoising trajectories projected onto F(s) (illustration)
L, lmax, T = 13, 8, 10; rng = np.random.default_rng(3); emb = lambda l: -1 + 2 * l / (L - 1); cap = emb(lmax)
targets = rng.choice([5, 6, 7, 8], 8, p=[0.2, 0.3, 0.3, 0.2])
fig, ax = plt.subplots(figsize=(3.0 * cm, 2.25 * cm), dpi=300)
ax.axhspan(cap, 1.15, color=R, alpha=0.13, lw=0); ax.axhline(cap, color=R, lw=0.6, ls="--")
ax.text(T, (cap + 1.12) / 2, "outside $\\mathcal{F}(s)$", color=R, ha="right", va="center", fontsize=4.6)
for j in range(8):
    x = rng.normal(0, 1); xs = [min(x, cap)]
    for t in range(1, T + 1):
        a = t / T; x = (1 - a) * x + a * emb(targets[j]) + rng.normal(0, 0.35 * (1 - a)); xs.append(min(x, cap))
    ax.plot(range(T + 1), xs, color=B, lw=0.55, alpha=0.8); ax.plot(T, xs[-1], "o", ms=1.5, color=B)
ax.set_xlim(0, T + 0.3); ax.set_ylim(-1.4, 1.15); ax.set_xticks([0, T]); ax.set_xticklabels([r"$x_T$", r"$x_0$"])
ax.set_xlabel("denoising step", labelpad=0.5)
ax.set_yticks([emb(0), emb(lmax), emb(12)]); ax.set_yticklabels(["0", r"$\ell_{\max}$", "12"]); ax.set_ylabel("level", labelpad=0.5)
clean(ax); fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.24); fig.savefig(f"{out}/denoise.pdf"); plt.close(fig)
# critic selection (illustration)
q = np.array([0.6, 1.1, 1.6, 2.1, 2.5, 2.8, 3.05, 3.2, 3.0, 2.2, 1.4, 0.7, 0.2])
samp = sorted(set(targets.tolist())); best = max(samp, key=lambda l: q[l])
fig, ax = plt.subplots(figsize=(2.6 * cm, 2.25 * cm), dpi=300)
col = [("0.85" if l > lmax else (B if l in samp else "#b9cbe6")) for l in range(L)]; col[best] = G
ax.bar(range(L), q, color=col, width=0.8); ax.axvline(lmax + 0.5, color=R, lw=0.6, ls="--")
ax.annotate("selected", xy=(best, q[best] + 0.05), xytext=(best - 6.5, q[best] + 0.35), color=G, fontsize=4.8,
            arrowprops=dict(arrowstyle="-|>", lw=0.5, color=G, mutation_scale=4))
ax.set_xticks([0, lmax, 12]); ax.set_xticklabels(["0", r"$\ell_{\max}$", "12"]); ax.set_xlabel("level", labelpad=0.5)
ax.set_yticks([]); ax.set_ylabel(r"$\bar Q(s,\ell)$", labelpad=1); ax.set_ylim(0, 4.0); clean(ax)
fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.24); fig.savefig(f"{out}/critic.pdf"); plt.close(fig)
# throughput sparkline (illustration)
tp = np.clip(3 + np.cumsum(rng.normal(0, 0.6, 24)), 0.6, None); tp[-5:] *= 0.45
fig, ax = plt.subplots(figsize=(1.6 * cm, 0.55 * cm), dpi=300)
ax.fill_between(range(24), tp, color=O, alpha=0.25, lw=0); ax.plot(tp, color=O, lw=0.6); ax.axis("off")
fig.subplots_adjust(0, 0, 1, 1); fig.savefig(f"{out}/tput.pdf"); plt.close(fig)
print("ok")
# ---- (c) conformal throughput bound (illustration)
rng2 = np.random.default_rng(7); t = np.arange(16); thr = np.clip(2.2 + 0.5 * np.sin(t / 2.5) + rng2.normal(0, 0.35, 16), 0.6, None)
thr[11:] *= 0.55
hm = np.array([len(thr[max(0, i - 5):i]) / np.sum(1 / thr[max(0, i - 5):i]) if i else thr[0] for i in range(16)])
fig, ax = plt.subplots(figsize=(3.0 * cm, 1.9 * cm), dpi=300)
ax.fill_between(t, 0, hm * 0.69, color=G, alpha=0.18, lw=0, step="mid")
ax.step(t, thr, where="mid", color=O, lw=0.8, label="realized $c_k$")
ax.step(t, hm, where="mid", color="0.35", lw=0.7, ls="--", label=r"forecast $\hat c_k$")
ax.step(t, hm * 0.69, where="mid", color=G, lw=0.9, label=r"bound $\underline{c}_k$")
miss = [i for i in range(16) if thr[i] < hm[i] * 0.69]
ax.plot(miss, thr[miss], "x", color=R, ms=2.5, mew=0.7)
ax.set_xticks([]); ax.set_yticks([0, 2]); ax.set_ylabel("Mbps", labelpad=0.5); ax.set_xlabel("chunk", labelpad=0.5); clean(ax)
fig.subplots_adjust(left=0.17, right=0.98, top=0.97, bottom=0.17); fig.savefig(f"{out}/conf.pdf"); plt.close(fig)
# ---- (c) look-ahead planning tree (illustration)
lmax = [7, 6, 6]; best = [6, 6, 5]
fig, ax = plt.subplots(figsize=(2.45 * cm, 2.3 * cm), dpi=300)
for j in range(3):
    ax.add_patch(plt.Rectangle((j - 0.32, -0.5), 0.64, lmax[j] + 1, color=G, alpha=0.16, lw=0))
    ax.add_patch(plt.Rectangle((j - 0.32, lmax[j] + 0.5), 0.64, 12 - lmax[j], color=R, alpha=0.08, lw=0))
for r in range(lmax[0] + 1):
    for d1 in (-1, 0, 1):
        l1 = r + d1
        if not 0 <= l1 <= lmax[1]: continue
        ax.plot([0, 1], [r, l1], color="0.7", lw=0.3)
        for d2 in (-1, 0, 1):
            l2 = l1 + d2
            if 0 <= l2 <= lmax[2]: ax.plot([1, 2], [l1, l2], color="0.8", lw=0.25)
ax.plot([0, 1, 2], best, "-o", color=B, lw=1.3, ms=2.5, zorder=5)
ax.scatter([0] * (lmax[0] + 1), range(lmax[0] + 1), s=3, color="0.4", zorder=4)
ax.annotate("best plan", xy=(1, best[1]), xytext=(0.6, 10.2), color=B, fontsize=4.8,
            arrowprops=dict(arrowstyle="-|>", lw=0.5, color=B, mutation_scale=4))
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["$k$", "$k{+}1$", "$k{+}2$"]); ax.set_ylim(-0.6, 12.5); ax.set_xlim(-0.45, 2.45)
ax.set_yticks([0, 6, 12]); ax.set_ylabel("level", labelpad=0.5); clean(ax)
fig.subplots_adjust(left=0.24, right=0.97, top=0.97, bottom=0.15); fig.savefig(f"{out}/plan.pdf"); plt.close(fig)
print("ok2")
