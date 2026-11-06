"""Panel (c) plot of Fig. 1: one real HSDPA session, DIPO vs. NeSyStream (levels + feasible set, stalls)."""
import json, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = json.load(open(sys.argv[1])); out = sys.argv[2]; vid, tr = "film_tears_of_steel_seg2", "norway_ferry_17"
get = lambda m: next(r for r in S[m] if r["video"] == vid and r["trace"] == tr)
o, d = get("Ours-plan3-all")["log"], get("DIPO-best8")["log"]
B, R, G, O = "#2f6db5", "#c8452f", "#3a9a5b", "#d0702a"; cm = 1 / 2.54; n = len(o); x = np.arange(n)
plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.45, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6,
                     "xtick.major.size": 1.8, "ytick.major.size": 1.8, "xtick.major.width": 0.4, "ytick.major.width": 0.4})
fig = plt.figure(figsize=(9.7 * cm, 3.45 * cm), dpi=300)
a0 = fig.add_axes([0.075, 0.76, 0.91, 0.19]); a1 = fig.add_axes([0.075, 0.37, 0.91, 0.35], sharex=a0); a2 = fig.add_axes([0.075, 0.17, 0.91, 0.16], sharex=a0)
thr = [e["thr"] for e in o]
a0.fill_between(x, thr, step="mid", color=O, alpha=0.3, lw=0); a0.step(x, thr, where="mid", color=O, lw=0.7)
a0.set_ylabel("Mbps", labelpad=1); a0.text(0.0, 1.0, "3G throughput", color=O, ha="left", va="bottom", fontsize=5.6, transform=a0.transAxes)
lm = np.array([e["l_max"] for e in o])
a1.bar(x, lm + 0.5, bottom=-0.5, width=1.0, color=G, alpha=0.18, lw=0, label=r"feasible set $\mathcal{F}(s)$")
a1.step(x, [e["level"] for e in d], where="mid", color=R, lw=0.9, label="DIPO")
a1.step(x, [e["level"] for e in o], where="mid", color=B, lw=1.1, label="NeSyStream (ours)")
v = [i for i, e in enumerate(d) if e["violated"]]
a1.plot(v, [d[i]["level"] for i in v], "x", ms=2.6, mew=0.7, color=R, label="rule violation")
a1.set_ylabel("level", labelpad=1); a1.set_ylim(-0.5, 13.5); a1.set_yticks([0, 4, 8, 12])
a1.legend(fontsize=5.2, ncol=4, loc="upper center", frameon=False, handlelength=1.2, columnspacing=0.8, labelspacing=0.1)
sd = [e["rebuf"] if i else 0 for i, e in enumerate(d)]; so = [e["rebuf"] if i else 0 for i, e in enumerate(o)]
a2.bar(x - 0.2, sd, width=0.4, color=R); a2.bar(x + 0.2, so, width=0.4, color=B)
a2.set_ylabel("stall s", labelpad=1); a2.set_yticks([0, 5]); a2.text(n - 0.4, -0.55, "chunk (2 s)", transform=a2.get_xaxis_transform(), ha="right", va="top", fontsize=5.4)
for a in (a0, a1):
    plt.setp(a.get_xticklabels(), visible=False)
for a in (a0, a1, a2):
    for s in ("top", "right"): a.spines[s].set_visible(False)
a2.set_xlim(-0.5, n - 0.5); a2.set_xticks([0, 5, 10, 15, 20, 25])
fig.savefig(out); print("violations", len(v), "stall DIPO", sum(sd), "ours", sum(so))
