"""Synthetic videos + traces for the smoke test (not for reported results)."""
import _common  # noqa
import argparse
from nesystream.data.synthetic import generate
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/synthetic")
ap.add_argument("--n_videos", type=int, default=8); ap.add_argument("--n_traces", type=int, default=20)
a = ap.parse_args(); generate(a.out, a.n_videos, a.n_traces); print("synthetic data in", a.out)
