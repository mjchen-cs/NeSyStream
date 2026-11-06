"""Train the 2026 baselines EAStream (context meta-RL) and NeuroBA (neuro-symbolic RL) on the same videos/traces.

python scripts/train_new_baselines.py --video_dir data/videos --trace_dir data/traces/train \
       --nsqoe runs/e2_sqoe3/weights_mos.json --bw_aug 0.3,600 --log_tput --out runs/final/newbase_s0 \
       --which eastream,neuroba --seed 0
Protocol identical to scripts/train_baselines.py: PQF observation (as PPO-VMAF), linear VMAF QoE reward,
NS-QoE energy table and l_max hidden, 200k decisions, small sweep, checkpoint chosen on validation traces
(every 10th training trace; test sets never used).
"""
import _common  # noqa
import argparse, itertools, json, os
import torch
from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
from nesystream.data.chunks import list_videos, split_by_video
from nesystream.models.nsqoe import NSQoE
from nesystream.eval.learned_baselines import reward_linear_vmaf
from nesystream.eval.new_baselines import (NBConfig, train_eastream, train_neuroba, eastream_act_fn, neuroba_act_fn,
                                          validation_score_fn, _layout)

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--trace_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--which", default="eastream,neuroba"); ap.add_argument("--steps", type=int, default=200000)
ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--playlist_chunks", type=int, default=30)
ap.add_argument("--rollout", type=int, default=256); ap.add_argument("--nsqoe", default=None)
ap.add_argument("--log_tput", action="store_true"); ap.add_argument("--bw_aug", default="0.3,60")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val_frac", type=float, default=0.1)
ap.add_argument("--lr", default="", help="override the learning-rate sweep (comma list)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tr, _, _ = split_by_video(list_videos(a.video_dir), os.path.join(a.video_dir, "split.json"))
videos = [load_video(p) for p in tr]; all_traces = load_traces(a.trace_dir)
k = max(2, int(round(1 / a.val_frac)))
val_traces = all_traces[::k]; traces = [t for i, t in enumerate(all_traces) if i % k]
W = json.load(open(a.nsqoe)) if a.nsqoe else None
print(f"traces: {len(traces)} train / {len(val_traces)} validation", flush=True)


def envs(trs=None, n=None):
    return [ABREnv(videos, trs or traces, NSQoE(init_weights=W), EnvConfig(quality_source="pqf", use_hard_rules=False,
                   playlist_chunks=a.playlist_chunks, bw_aug=a.bw_aug, log_tput=a.log_tput), seed=a.seed * 100 + i)
            for i in range(n or a.n_envs)]


rf = lambda r, p, e: reward_linear_vmaf(r, p)
E0 = envs(n=1)[0]; mask = E0.L + 1
# sweeps: EAStream adapts its entropy weight (sweep lr only, paper value 1e-4 included); NeuroBA: ent x lr as the others
SWEEP = {"eastream": [dict(lr=lr) for lr in ([float(x) for x in a.lr.split(",")] if a.lr else [1e-4, 3e-4])],
         "neuroba": [dict(ent=e_, lr=lr) for e_, lr in itertools.product([0.01, 0.05],
                     [float(x) for x in a.lr.split(",")] if a.lr else [3e-4, 1e-3])]}
for name in a.which.split(","):
    sweep, best = [], None
    for hp in SWEEP[name]:
        print(f"=== training {name} {hp}", flush=True); torch.manual_seed(a.seed)
        cfg = NBConfig(total_steps=a.steps, n_envs=a.n_envs, rollout=a.rollout, mask_tail=mask, **hp)
        V = envs(trs=val_traces, n=1)
        if name == "eastream":
            vf = lambda net: validation_score_fn(eastream_act_fn(net, rf), V, rf)
            net = train_eastream(envs(), rf, cfg, val_fn=vf)
        else:
            vf = lambda net: validation_score_fn(neuroba_act_fn(net), V, rf)
            net = train_neuroba(envs(), rf, cfg, val_fn=vf)
        sweep.append(dict(hp, val_score=net.val_score)); print(sweep[-1], flush=True)
        if best is None or net.val_score > best[0]:
            best = (net.val_score, {k_: v_.detach().cpu().clone() for k_, v_ in net.state_dict().items()}, hp, cfg)
    cfg = best[3]
    ck = {"kind": name, "state": best[1], "obs_dim": E0.obs_dim, "L": E0.L, "quality_source": "pqf", "mask_tail": mask,
          "log_tput": a.log_tput, "val_score": best[0], "hp": best[2], "sweep": sweep}
    if name == "eastream":
        ck["m_dim"] = cfg.m_dim
    else:
        ck.update(layout=_layout(E0), n_rules=cfg.n_rules, n_thr=cfg.n_thr)
    torch.save(ck, os.path.join(a.out, f"{name}.ptn"))
    json.dump(sweep, open(os.path.join(a.out, f"{name}_sweep.json"), "w"), indent=1)
    print(f"*** {name}: selected {best[2]} (validation {best[0]:.3f})", flush=True)
