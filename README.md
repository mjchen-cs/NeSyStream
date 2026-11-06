# NeSyStream

Code for the paper *Seeing Before Streaming: Neuro-Symbolic Planning with Conformal Safety for Perception-Aware
Adaptive Video Streaming* (anonymous submission).

NeSyStream has three parts:
1. **Perceptual Quality Field (PQF)** - the server measures VMAF on two probe renditions of every chunk and completes the
   full rate-quality curve with a monotone head; interpretable detectors give five visual concepts.
2. **Neuro-Symbolic QoE (NS-QoE)** - QoE as weighted first-order rules over visual and playback predicates; soft-rule
   weights are fit to subjective scores, hard rules define the admissible actions.
3. **Conformal feasible set + rule-based look-ahead planning** - the throughput bound of the buffer-safety rule is
   calibrated with (adaptive) conformal inference, and a short look-ahead over the admissible levels maximises NS-QoE.

## Requirements

```bash
pip install -r requirements.txt        # torch, numpy, itu-p1203; optional: opencv-python
```
`ffmpeg`/`ffprobe` with `libvmaf` must be on the PATH (`ffmpeg -filters | grep libvmaf`).
The dash.js deployment additionally needs Node.js and Playwright (`npm i playwright`).

## Data (all public)

| Data | Use | Source |
|---|---|---|
| Xiph derf, Netflix Open Content (El Fuente, Meridian, Sparks, Cosmos Laundromat), YouTube-UGC, UVG, Tears of Steel, Big Buck Bunny | source videos | see `scripts/prepare_sources.py` |
| Puffer | training traces | puffer.stanford.edu |
| HSDPA (3G), FCC, Oboe, Lumos 4G / 5G | test traces | see `scripts/prepare_traces.py` |
| Waterloo SQoE-III | NS-QoE fitting (MOS) | Duanmu et al., IEEE T-BC 2018 |
| LIVE-NFLX-II | zero-shot QoE validation only | Bampis et al., IEEE TIP 2021 |

A synthetic dataset for a quick end-to-end check (numbers are meaningless) is produced by `scripts/smoke_test.sh`.

## Pretrained models and data

`assets/` contains the per-chunk descriptors of all 114 sources, the trained checkpoints of NeSyStream's learned
components and of all learned baselines (3 seeds), the conformal calibration, the NS-QoE weights, and the results
and per-session logs of every evaluation in the paper. See [`assets/README.md`](assets/README.md) for the contents
and a one-command reproduction of the end-to-end table.

## Pipeline

```bash
# 1. traces and sources -> uniform reference clips -> 2-s chunks, 13-rung ladder (<=1080p), per-chunk VMAF, concepts
python scripts/prepare_traces.py  --zip_dir <trace_zips> --out data/traces
python scripts/prepare_sources.py --root <downloads> --out data/refs
python scripts/build_dataset.py   --src_dir data/refs --out_dir data/videos --chunk_len 2 --keep_renditions

# 2. PQF (Sec. 4.1, Tab. "rate-quality"): probe-and-complete outputs cached per chunk; learned/VQA baselines for comparison
python scripts/cache_quality.py --video_dir data/videos
python scripts/train_feature_baseline.py --video_dir data/videos --kind handcrafted --out runs/e1_handcrafted.json

# 3. NS-QoE (Sec. 4.2, Tab. "QoE"): fit rule weights on SQoE-III MOS with held-out contents; bootstrap intervals
python scripts/sqoe3_to_sessions.py --root <sqoe3_root> --out data/sqoe3/sqoe3_sessions.npz --concepts data/sqoe3/sqoe3_concepts.json
python scripts/cv_sqoe3.py --sessions data/sqoe3/sqoe3_sessions.npz --root <sqoe3_root> --out runs/e2_sqoe3
python scripts/bootstrap_nsqoe_sqoe3.py

# 4. conformal calibration of the H1 throughput bound (training traces only)
python scripts/calibrate_conformal.py --video_dir data/videos --trace_dir data/traces/train --out runs/conformal/ratios.json

# 5. learned components and baselines (see BASELINES.md)
python scripts/train_policy.py --video_dir data/videos --trace_dir data/traces/train --nsqoe runs/e2_sqoe3/weights_mos.json \
       --bw_aug 0.3,600 --log_tput --out runs/final/ours_s0

# 6. end-to-end evaluation (Tab. "end-to-end"); NeSyStream = Ours-plan3-all with the conformal bound
for T in hsdpa fcc oboe lumos4g lumos5g; do
  python scripts/evaluate.py --video_dir data/videos --trace_dir data/traces/test/$T --policy runs/final/ours_s0 \
         --trace_sample 60 --plan 3 --hard_tput conformal --conf_file runs/conformal/ratios.json --conf_alpha 0.05 \
         --baselines_dir runs/final/baselines_s0 --out runs/final/eval_$T.json
done

# 7. cross-dataset QoE validation and metric consistency (supplement)
python scripts/nflx2_eval.py --json data/live_nflx2/nflx2_segments.json --out runs/e2_nflx2
python scripts/qoe_metric_consistency.py --sessions runs/e3_sessions_compact.json --out runs/e3_metrics
```
Every script documents its options in its header and with `--help`. Hyper-parameter selection uses validation videos and
training traces only (`evaluate.py --split val`); the test sets are evaluated once.

## Real-player deployment (supplement)

`deploy_dashjs/`: dash.js player in headless Chromium with a custom ABR rule that queries a local decision service
(`abr_server.py`, same PQF / NS-QoE / conformal / planner code), a trace-shaped HTTP segment server (`seg_server.py`),
the session harness (`run_sessions.js`) and the analysis (`analyze.py`).

## Layout

| Path | Content |
|---|---|
| `nesystream/data/` | dataset construction, chunk features, statistics |
| `nesystream/models/pqf.py` | PQF (probe-and-complete, monotone head, concept detectors, learned variants) |
| `nesystream/models/nsqoe.py` | NS-QoE predicates, rules, learned weights, rule induction, feasible set |
| `nesystream/env/abr_env.py` | trace-driven simulator, conformal H1 bound with ACI |
| `nesystream/eval/` | heuristic, learned and diffusion baselines; metrics incl. ITU-T P.1203 |
| `nesystream/models/diffusion_policy.py`, `nesystream/rl/` | feasible-set diffusion policy (baseline / ablation) |
| `scripts/` | all experiments and figures |
| `deploy_dashjs/` | dash.js deployment |
