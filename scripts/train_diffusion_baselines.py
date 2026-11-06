"""Train the NeurIPS 2025 diffusion-RL baselines GenPO (on-policy, flow policy) and VDRL (value diffusion).

python scripts/train_diffusion_baselines.py --video_dir data/videos --trace_dir data/traces/train \
       --nsqoe runs/e2_sqoe3/weights_mos.json --bw_aug 0.3,600 --log_tput --steps 300000 \
       --which genpo,vdrl --seed 0 --out runs/final/diffbase_s0
Same protocol as DIPO / QSM (environment reward = NS-QoE fitted to MOS, no hard rules, 300k decisions, PQF
observation) with the NS-QoE energy table and l_max hidden; the checkpoint is chosen on validation traces
(every 10th training trace). Paper hyper-parameters, no sweep (as DIPO / QSM).
"""
import _common  # noqa
import argparse, json, os
import torch
from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
from nesystream.data.chunks import list_videos, split_by_video
from nesystream.models.nsqoe import NSQoE
from nesystream.eval.diffusion_baselines import (DBConfig, train_genpo, train_vdrl, genpo_act_fn, vdrl_act_fn,
                                                env_reward_validation)

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--trace_dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--which", default="genpo,vdrl"); ap.add_argument("--steps", type=int, default=300000)
ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--playlist_chunks", type=int, default=30)
ap.add_argument("--nsqoe", default=None); ap.add_argument("--log_tput", action="store_true")
ap.add_argument("--bw_aug", default="0.3,60"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.1); ap.add_argument("--val_every", type=int, default=20000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
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


E0 = envs(n=1)[0]; mask = E0.L + 1
for name in a.which.split(","):
    torch.manual_seed(a.seed)
    cfg = DBConfig(total_steps=a.steps, n_envs=a.n_envs, mask_tail=mask, val_every=a.val_every)
    V = envs(trs=val_traces, n=1)
    print(f"=== training {name}  {cfg}", flush=True)
    if name == "genpo":
        net = train_genpo(envs(), cfg, val_fn=lambda n_: env_reward_validation(genpo_act_fn(n_), V))
        extra = {"flow_steps": cfg.flow_steps, "mix": cfg.mix}
    else:
        net = train_vdrl(envs(), cfg, val_fn=lambda n_: env_reward_validation(vdrl_act_fn(n_), V))
        extra = {"T": cfg.T}
    torch.save({"kind": name, "state": {k_: v_.detach().cpu() for k_, v_ in net.state_dict().items()}, "obs_dim": E0.obs_dim,
                "L": E0.L, "quality_source": "pqf", "mask_tail": mask, "log_tput": a.log_tput, "val_score": net.val_score, **extra},
               os.path.join(a.out, f"{name}.ptd"))
    print(f"*** {name}: validation {net.val_score:.3f}", flush=True)
