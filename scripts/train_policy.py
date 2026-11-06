"""Train the logic-guided diffusion policy (+ consistency distillation).

python scripts/train_policy.py --video_dir data/videos --trace_dir data/traces/train \
       --nsqoe runs/nsqoe.json --out runs/policy
Ablations: --quality_source {pqf,oracle,bitrate}  --no_hard_rules  --eta_guid 0  --T 100  --n_q 2
"""
import _common  # noqa
import argparse, json, os
import torch
from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
from nesystream.data.chunks import list_videos, split_by_video
from nesystream.models.nsqoe import NSQoE
from nesystream.rl.trainer import TrainConfig, train, distill

ap = argparse.ArgumentParser()
ap.add_argument("--video_dir", required=True); ap.add_argument("--trace_dir", required=True)
ap.add_argument("--nsqoe", default=None); ap.add_argument("--out", required=True)
ap.add_argument("--quality_source", default="pqf", choices=["pqf", "pqf_cheap", "oracle", "bitrate"])
ap.add_argument("--no_hard_rules", action="store_true")
ap.add_argument("--no_concepts", action="store_true", help="ablation: the policy sees no visual concepts")
ap.add_argument("--log_tput", action="store_true", help="observe throughput on a log scale (needed for 5G)")
ap.add_argument("--bw_aug", default="0.3,60", help="training throughput augmentation lo,hi (Mbps, log-uniform); '' = off")
ap.add_argument("--steps", type=int, default=200000); ap.add_argument("--warmup", type=int, default=5000)
ap.add_argument("--warmup_policy", default="random", choices=["random", "mpc"])
ap.add_argument("--T", type=int, default=20); ap.add_argument("--n_q", type=int, default=10)
ap.add_argument("--eta_guid", type=float, default=1.0); ap.add_argument("--ent", type=float, default=0.1)
ap.add_argument("--n_envs", type=int, default=8); ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--playlist_chunks", type=int, default=30, help="session length in chunks (0 = one clip per session)")
ap.add_argument("--policy", default="diffusion", choices=["diffusion", "categorical"])
ap.add_argument("--reward", default="nsqoe", choices=["nsqoe", "linear_vmaf"])
ap.add_argument("--actor_update", default="aw", choices=["aw", "dipo", "qsm"])
ap.add_argument("--distill_steps", type=int, default=20000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
os.makedirs(a.out, exist_ok=True)
tr, _, _ = split_by_video(list_videos(a.video_dir), os.path.join(a.video_dir, "split.json"))
videos = [load_video(p) for p in tr]
traces = load_traces(a.trace_dir)
w = json.load(open(a.nsqoe)) if a.nsqoe else None
q = NSQoE(init_weights=w)
cfg_env = EnvConfig(quality_source=a.quality_source, use_hard_rules=not a.no_hard_rules, reward=a.reward,
                    playlist_chunks=a.playlist_chunks, bw_aug=a.bw_aug, use_concepts=not a.no_concepts, log_tput=a.log_tput)
envs = [ABREnv(videos, traces, q, cfg_env, seed=a.seed * 100 + i) for i in range(a.n_envs)]
cfg = TrainConfig(total_steps=a.steps, warmup=a.warmup, T=a.T, n_q=a.n_q, eta_guid=a.eta_guid, ent=a.ent,
                  n_envs=a.n_envs, batch=a.batch, actor_update=a.actor_update, policy=a.policy,
                  warmup_policy=a.warmup_policy)
agent, rb = train(envs, cfg)
torch.save({"pi": agent.pi.state_dict(), "q": agent.q.state_dict(), "obs_dim": envs[0].obs_dim, "L": envs[0].L,
            "cfg": vars(cfg), "env": vars(cfg_env), "nsqoe": w}, os.path.join(a.out, "teacher.pt"))
if a.distill_steps > 0 and a.policy == "diffusion":
    st = distill(agent, rb, a.distill_steps)
    torch.save({"student": st.state_dict()}, os.path.join(a.out, "student.pt"))
print("saved to", a.out)
