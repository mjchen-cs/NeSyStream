#!/usr/bin/env bash
# End-to-end smoke test on SYNTHETIC data (checks that every stage runs; numbers are meaningless).
set -e
cd "$(dirname "$0")/.."
R=runs/smoke; D=data/synthetic
mkdir -p $R
python scripts/make_synthetic.py --out $D --n_videos 8 --n_traces 12
python - <<'PY'
import glob, json, os
p = sorted(glob.glob("data/synthetic/videos/*.npz")); n = [os.path.splitext(os.path.basename(x))[0] for x in p]
json.dump({"train": n[:5], "val": n[5:6], "test": n[6:]}, open("data/synthetic/videos/split.json", "w"))
PY
python scripts/train_pqf.py --video_dir $D/videos --out $R/pqf.pt --epochs 3 --bs 32 --width 16
python scripts/cache_pqf.py --video_dir $D/videos --ckpt $R/pqf.pt
python scripts/make_sessions.py --video_dir $D/videos --trace_dir $D/traces --out $R/sessions.npz --n 300 --target synthetic
python scripts/fit_nsqoe.py --sessions $R/sessions.npz --out $R/nsqoe.json --steps 1500
python scripts/train_policy.py --video_dir $D/videos --trace_dir $D/traces --nsqoe $R/nsqoe.json --out $R/policy \
       --steps 3000 --warmup 500 --n_envs 4 --batch 64 --n_q 4 --distill_steps 300
python scripts/evaluate.py --video_dir $D/videos --trace_dir $D/traces --policy $R/policy --out $R/eval.json --max_traces 3
echo "SMOKE TEST PASSED"
