# Baselines (re-implemented on the NeSyStream simulator)

All baselines share the same videos, 13-level ladder, training traces (Puffer, bw_aug 0.3-600 Mbps, log throughput),
evaluation sessions (5 test sets, 60 sampled traces, fixed seeds) and checkpoint selection on validation traces.
The NS-QoE energy table and l_max (our symbolic layer) are hidden from every learned baseline.

| Method | Paper | Code | Train script | Checkpoints |
|---|---|---|---|---|
| BBA, RB, RobustMPC | classic | nesystream/eval/baselines.py | - | - |
| Pensieve-PPO, PPO-VMAF, Jade, PPO-Lagrangian | SIGCOMM'17, ACM MM'23, safe RL | nesystream/eval/learned_baselines.py | scripts/train_baselines.py | runs/final/baselines_s{0,1,2}/*.pt |
| DIPO, QSM, feasible-set diffusion | diffusion RL | nesystream/rl/trainer.py, nesystream/models/diffusion_policy.py | scripts/train_policy.py | runs/final/{dipo,qsm,ours}_s* |
| EAStream | IEEE TSC 19(2), 2026 | nesystream/eval/new_baselines.py | scripts/train_new_baselines.py --which eastream | runs/final/newbase_s*/eastream.ptn |
| NeuroBA | IEEE ToN 34, 2026 | nesystream/eval/new_baselines.py | scripts/train_new_baselines.py --which neuroba | runs/final/newbase_s*/neuroba.ptn |
| GenPO | NeurIPS 2025 (official code github.com/wadx2019/genpo, flow ported) | nesystream/eval/diffusion_baselines.py | scripts/train_diffusion_baselines.py --which genpo | runs/final/diffbase_s*/genpo.ptd |
| VDRL | NeurIPS 2025 | nesystream/eval/diffusion_baselines.py | scripts/train_diffusion_baselines.py --which vdrl | runs/final/diffbase_s*/vdrl.ptd |

Evaluation: scripts/evaluate.py --baselines_dir (PPO family), --newbase_dir (EAStream/NeuroBA), --diffbase_dir (GenPO/VDRL),
--extra_policies (DIPO/QSM). Example (job 480):
  python scripts/evaluate.py --video_dir data/videos --trace_dir data/traces/test/hsdpa --policy runs/final/ours_s0
         --out runs/final/eval_newbase_s0_hsdpa.json --jobs 12 --trace_sample 60 --only EAStream,NeuroBA,GenPO,VDRL
         --newbase_dir runs/final/newbase_s0 --diffbase_dir runs/final/diffbase_s0

Adaptations (details in the module docstrings and the paper supplement):
- EAStream: GRU belief encoder + VariBAD-style decoder (ELBO, lambda_s=lambda_r=1, beta=0.1, m=16), PPO policy pi(a|s,m),
  automatic entropy weight (target 0.1); linear VMAF reward (as PPO-VMAF), 200k decisions.
- NeuroBA: single client, no IRS / channel-gain action; predicates = learned thresholds on per-level and global features,
  LNN weighted AND/OR neurons (weights >= 0 by projection), PPO-clip + prioritized replay; linear VMAF reward, 200k.
- GenPO: official flow (doubled dummy action, exact inverse, log|det J|), official Ant hyper-parameters, KL-adaptive lr;
  level = embed_to_level((z+y)/2); NS-QoE reward (as DIPO/QSM), 300k.
- VDRL: two conditional DDPMs (T=10) over the return, target r + gamma*min_i Z_i(s',a'); categorical policy maximising
  sum_a pi(a|s) min_i E[Z_i(s,a)] (+0.01 entropy); NS-QoE reward, 300k.
