"""Train PPO-family baselines on the same videos/traces.

python scripts/train_baselines.py --video_dir data/videos --trace_dir data/traces/train \
       --sessions runs/sessions.npz --out runs/baselines --which pensieve,ppo_vmaf,jade,ppo_lag
Diffusion-RL baselines (DIPO, QSM) are trained with train_policy.py:
  --actor_update dipo --eta_guid 0 --no_hard_rules      (same visual input, no symbolic layer)
"""
import _common  # noqa
import argparse, json, os
import torch
from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
from nesystream.data.chunks import list_videos, split_by_video
from nesystream.models.nsqoe import NSQoE
from nesystream.eval.learned_baselines import (PPOConfig, train_ppo, reward_linear_bitrate, reward_linear_vmaf,
                                               fit_rank_qoe)

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--trace_dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--sessions", default=None, help="needed for jade")
ap.add_argument("--which", default="pensieve,ppo_vmaf,jade,ppo_lag")
ap.add_argument("--steps", type=int, default=200000); ap.add_argument("--n_envs", type=int, default=8)
ap.add_argument("--playlist_chunks", type=int, default=30)
ap.add_argument("--rollout", type=int, default=256)
ap.add_argument("--nsqoe", default=None, help="NS-QoE weights json (same reward as ours, e.g. the MOS-fitted one)")
ap.add_argument("--log_tput", action="store_true", help="throughput observed on a log scale (as ours)")
ap.add_argument("--bw_aug", default="0.3,60", help="training throughput augmentation lo,hi (Mbps, log-uniform); '' = off")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_mask_obs", action="store_true",
                help="let the baselines see our NS-QoE energy table and l_max (default: hidden)")
ap.add_argument("--ent", default="0.01,0.05", help="entropy coefficients to sweep")
ap.add_argument("--lr", default="3e-4,1e-4", help="learning rates to sweep")
ap.add_argument("--val_frac", type=float, default=0.1, help="training traces held out for model selection")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
tr, _, _ = split_by_video(list_videos(a.video_dir), os.path.join(a.video_dir, "split.json"))
videos = [load_video(p) for p in tr]; all_traces = load_traces(a.trace_dir)
# validation traces: every k-th TRAINING trace (the test sets are never used for tuning)
k = max(2, int(round(1 / a.val_frac)))
val_traces = all_traces[::k]; traces = [t for i, t in enumerate(all_traces) if i % k]
W = json.load(open(a.nsqoe)) if a.nsqoe else None
print(f"traces: {len(traces)} train / {len(val_traces)} validation")

def envs(src, hard=False, trs=None, n=None):
    return [ABREnv(videos, trs or traces, NSQoE(init_weights=W), EnvConfig(quality_source=src, use_hard_rules=hard,
                   playlist_chunks=a.playlist_chunks, bw_aug=a.bw_aug, log_tput=a.log_tput), seed=a.seed * 100 + i)
            for i in range(n or a.n_envs)]

from nesystream.eval.learned_baselines import validation_score
import itertools, json
cfg = lambda **k: PPOConfig(total_steps=a.steps, n_envs=a.n_envs, rollout=a.rollout, **k)
jobs = {
    "pensieve": ("bitrate", lambda r, p, e: reward_linear_bitrate(r, p), {}),
    "ppo_vmaf": ("pqf", lambda r, p, e: reward_linear_vmaf(r, p), {}),
    "ppo_lag": ("pqf", lambda r, p, e: reward_linear_vmaf(r, p) + 4.3 * r["rebuf"], {"lagrangian": True}),
}
if "jade" in a.which:
    assert a.sessions, "--sessions (with targets) required for the Jade-style ranking QoE"
    rq = fit_rank_qoe(a.sessions)
    def jade_r(r, p, e):
        T = lambda x: torch.tensor([float(x)])
        with torch.no_grad():
            return float(rq.chunk(T(r["vmaf"]), T(r["vmaf"] if p is None else p["vmaf"]), T(r["rebuf"]),
                                  T(r["level"] / (e.L - 1)))[0])
    jobs["jade"] = ("pqf", jade_r, {})
for name in a.which.split(","):
    src, rf, extra = jobs[name]
    E = envs(src); V = envs(src, trs=val_traces, n=1)
    mask = 0 if a.no_mask_obs else E[0].L + 1          # energy table (L) + l_max (1) are the last entries
    sweep, best = [], None
    for ent, lr in itertools.product([float(x) for x in a.ent.split(",")], [float(x) for x in a.lr.split(",")]):
        print(f"=== training {name}  ent={ent} lr={lr}  mask_tail={mask}", flush=True)
        torch.manual_seed(a.seed)
        c = cfg(ent=ent, lr=lr, mask_tail=mask, **extra)
        vf = lambda net, c=c: validation_score(net, V, rf, lagrangian=c.lagrangian, cost_limit=c.cost_limit)
        net = train_ppo(envs(src), rf, c, val_fn=vf)
        sweep.append({"ent": ent, "lr": lr, "val_score": net.val_score}); print(sweep[-1], flush=True)
        if best is None or net.val_score > best[0]:
            best = (net.val_score, {k_: v_.detach().cpu().clone() for k_, v_ in net.state_dict().items()}, ent, lr)
    torch.save({"state": best[1], "obs_dim": E[0].obs_dim, "L": E[0].L, "quality_source": src, "mask_tail": mask, "log_tput": a.log_tput,
                "val_score": best[0], "ent": best[2], "lr": best[3], "sweep": sweep},
               os.path.join(a.out, f"{name}.pt"))
    json.dump(sweep, open(os.path.join(a.out, f"{name}_sweep.json"), "w"), indent=1)
    print(f"*** {name}: selected ent={best[2]} lr={best[3]} (validation {best[0]:.3f})", flush=True)
