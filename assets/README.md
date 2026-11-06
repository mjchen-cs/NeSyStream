# Released models, descriptors and results

Everything needed to reproduce the tables of the paper without re-encoding the videos or re-training.

| Folder | Content | Size |
|---|---|---|
| `descriptors/` | One `.npz` per source (114: 72 train / 14 val / 28 test, see `split.json`): chunk sizes and VMAF of all 13 rungs, bitrate ladder, visual concepts, and the PQF output (`pqf_q`: completed rate-quality curve, `pqf_z`: probe descriptor). This is the input of the simulator (`nesystream.env.abr_env.load_video`); no video frames are included. | <1 MB |
| `checkpoints/` | Trained policies, 3 seeds where the paper reports 3. `ours_s{0,1,2}`: critic and diffusion policy used for the feasible-set diffusion variant and for root proposals (the rule-based planner itself has no trainable policy); `baselines_s*`: Pensieve, PPO-VMAF, Jade, PPO-Lagrangian (+ learning-rate / entropy sweeps); `newbase_s*`: EAStream, NeuroBA; `diffbase_s*`: VDRL; `dipo_s*`, `qsm_s0`: DIPO, QSM; `abl_*`, `nohard_s0`: ablations. | 72 MB |
| `calibration/` | `ratios.json`: conformal calibration ratios of the H1 throughput bound (training traces only); `weights_mos.json`: NS-QoE rule weights fitted to SQoE-III MOS. | <1 MB |
| `results/` | Summary metrics of every evaluation in the paper (`final/`, `final2/`), the bootstrap intervals of the quality-vs-stalls figure (`pareto_ci.json`), and per-session logs (`sessions_*.tar.gz`). | 29 MB |
| `traces/` | Exact file lists of the training traces and of the five test sets. | <1 MB |

Network traces, source videos and subjective datasets are third-party data and are not redistributed; `scripts/prepare_traces.py` and the download scripts in `scripts/` rebuild them, and `traces/*.txt` lists the files used.

## Reproduce the end-to-end table from the released files

```bash
for T in hsdpa fcc oboe lumos4g lumos5g; do
  python scripts/evaluate.py --video_dir assets/descriptors --trace_dir data/traces/test/$T \
         --policy assets/checkpoints/ours_s0 --trace_sample 60 --plan 3 \
         --hard_tput conformal --conf_file assets/calibration/ratios.json --conf_alpha 0.05 --conf_gamma 0.01 \
         --baselines_dir assets/checkpoints/baselines_s0 --out runs/repro/eval_$T.json
done
```
Results should match `results/final2/eval_<set>.json` (method `Ours-plan3-all`).
