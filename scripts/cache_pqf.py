"""Run the trained PQF once per chunk (server side) and store q_hat, z, concepts in each video npz."""
import _common  # noqa
import argparse
import numpy as np, torch
from nesystream.data.chunks import list_videos, ChunkDataset
from nesystream.models.pqf import PQF

ap = argparse.ArgumentParser(); ap.add_argument("--video_dir", required=True); ap.add_argument("--ckpt", required=True)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu")
m = PQF(ck["L"], ck["C"], ck["D"], ck["z_dim"], ck["width"], backbone=ck.get("backbone", "scratch"), pretrained=False,
        inputs=ck.get("inputs", "both"), teacher_dim=ck.get("teacher_dim", 0), pair={'fragpair': 1, 'fragtri': 2}.get(ck.get("frames"), 0))
m.load_state_dict(ck["state"]); m.eval()
mu, sd = (np.array(x, np.float32) for x in ck["cmp_stats"])
for p in list_videos(a.video_dir):
    d = dict(np.load(p, allow_pickle=False))
    ds = ChunkDataset([p], (mu, sd), frames=ck.get("frames", "thumb"), anchors=ck.get("anchors", False))
    with torch.no_grad():
        o = m(torch.from_numpy(ds.frames), torch.from_numpy(ds.cmp))
    d["pqf_q"] = o["q"].numpy(); d["pqf_z"] = o["z"].numpy(); d["pqf_concepts"] = torch.sigmoid(o["concept_logits"]).numpy()
    np.savez_compressed(p, **d)
    print(p, "mae vs label %.2f" % np.abs(d["pqf_q"] - d["vmaf"]).mean())
