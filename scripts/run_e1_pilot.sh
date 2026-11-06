#!/usr/bin/env bash
# E1 pilot (10-20 sources is enough): PQF variants vs. simple and deep VQA baselines, then the Pareto figure.
# Usage: bash scripts/run_e1_pilot.sh data/videos runs/e1   [FineVQ npz files are imported beforehand, see third_party/finevq]
set -e
cd "$(dirname "$0")/.."
V=${1:-data/videos}; R=${2:-runs/e1}; EP=${EPOCHS:-100}
mkdir -p $R
# --- ours ---
python scripts/train_pqf.py --video_dir $V --out $R/pqf_scratch_32.pt --epochs $EP
python scripts/train_pqf.py --video_dir $V --out $R/pqf_scratch_16.pt --epochs $EP --z_dim 16
python scripts/train_pqf.py --video_dir $V --out $R/pqf_scratch_64.pt --epochs $EP --z_dim 64
python scripts/train_pqf.py --video_dir $V --out $R/pqf_mnv3_32.pt     --epochs $EP --backbone mobilenet_v3_small
python scripts/train_pqf.py --video_dir $V --out $R/pqf_bitstream.pt   --epochs $EP --inputs bitstream
python scripts/train_pqf.py --video_dir $V --out $R/pqf_frames.pt      --epochs $EP --inputs frames
# --- baselines ---
python scripts/train_feature_baseline.py --video_dir $V --kind bitrate     --out $R/base_bitrate.json
python scripts/train_feature_baseline.py --video_dir $V --kind handcrafted --out $R/base_handcrafted.json
for F in finevq dover fastvqa; do
  if ls $V/*_${F}.npy >/dev/null 2>&1; then
    python scripts/train_feature_baseline.py --video_dir $V --kind features --feat_name $F --out $R/base_${F}_feat.json
  fi
  if ls $V/*_${F}_scores.npy >/dev/null 2>&1; then
    python scripts/train_feature_baseline.py --video_dir $V --kind per_rendition --feat_name $F --out $R/base_${F}_all.json
  fi
done
# --- optional: offline distillation from FineVQ (run only if FineVQ features beat PQF clearly) ---
if [ "${DISTILL:-0}" = "1" ]; then
  python scripts/train_pqf.py --video_dir $V --out $R/pqf_distill_finevq.pt --epochs $EP --teacher finevq
fi
python - <<PY
import glob, json
rows = []
for p in sorted(glob.glob("$R/*_metrics.json")):
    r = json.load(open(p)); rows.append((p.split("/")[-1][:-13], r["test"]["mae"], r["test"]["srocc"], r["test"]["srocc_content"], r["test"]["bytes_est"], r["cost"]["ms_per_chunk"]))
for p in sorted(glob.glob("$R/base_*.json")):
    r = json.load(open(p)); rows.append((r["name"], r["mae"], r["srocc"], r["srocc_content"], r["cost"]["bytes_per_chunk"], r["cost"]["ms_per_chunk"]))
print(f"{'method':32s} {'MAE':>7s} {'SROCC':>7s} {'SROCC_c':>8s} {'bytes':>8s} {'ms':>9s}")
for n, *v in rows: print(f"{n:32s} " + " ".join(f"{x:>8.3f}" for x in v))
PY
python scripts/plot_e1_pareto.py --results $R/*_metrics.json $R/base_*.json --out $R/e1_pareto.pdf
