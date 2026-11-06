"""Build the chunk-level rate-quality dataset from source videos with ffmpeg.

For every source video:
  1. encode the bitrate ladder (forced key frames every `chunk_len` seconds);
  2. split each rendition into chunks with stream copy -> chunk sizes (bytes);
  3. compute per-frame quality vs. the source (libvmaf if available, else PSNR as a
     clearly-labelled fallback) and average per chunk -> rate-quality curves;
  4. decode the LOWEST rendition at low resolution -> keyframes + SI/TI, motion,
     scene-cut, luma, face statistics; read packet sizes with ffprobe -> compressed-domain stats.
Concept soft labels are normalised across the whole dataset in `finalize_concepts`.

Requires ffmpeg/ffprobe in PATH (with --enable-libvmaf for VMAF).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np

CONCEPT_FEATS = ["ti", "si", "scene", "face", "luma"]  # raw features -> VISUAL predicates


def run(cmd: List[str], capture=False, cwd=None):
    r = subprocess.run(cmd, stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=False, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{r.stderr.decode(errors='ignore')[-2000:]}")
    return r.stdout


def has_libvmaf() -> bool:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " libvmaf " in out


def probe(src):
    out = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames:format=duration", "-of", "json", src],
                         capture=True))
    st = out["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return int(st["width"]), int(st["height"]), float(num) / float(den), float(out["format"]["duration"])


def encode_ladder(src, out_dir, ladder: List[Tuple[int, int]], chunk_len: float, codec="libx264",
                  preset="medium", max_seconds=None, threads=0):
    """ladder: list of (bitrate_kbps, height). Returns list of rendition paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (kbps, h) in enumerate(ladder):
        p = os.path.join(out_dir, f"level_{i:02d}.mp4")
        cmd = ["ffmpeg", "-y", "-i", src]
        if max_seconds:
            cmd += ["-t", str(max_seconds)]
        cmd += ["-an", "-vf", f"scale=-2:{h}:flags=bicubic", "-c:v", codec, "-preset", preset, "-threads", str(threads),
                "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.2)}k", "-bufsize", f"{int(kbps * 2)}k",
                "-force_key_frames", f"expr:gte(t,n_forced*{chunk_len})", "-sc_threshold", "0"]
        if codec == "libx265":
            cmd += ["-x265-params", "log-level=error"]
        run(cmd + [p])
        paths.append(p)
    return paths


def chunk_sizes(rendition, chunk_len, tmp):
    d = tempfile.mkdtemp(dir=tmp)
    run(["ffmpeg", "-y", "-i", rendition, "-c", "copy", "-f", "segment", "-segment_time", str(chunk_len),
         "-reset_timestamps", "1", os.path.join(d, "c_%05d.mp4")])
    files = sorted(f for f in os.listdir(d) if f.endswith(".mp4"))
    sizes = [os.path.getsize(os.path.join(d, f)) for f in files]
    shutil.rmtree(d)
    return np.array(sizes, np.int64)


def per_frame_quality(rendition, src, w, h, tmp, metric="vmaf", max_seconds=None, threads=8, subsample=1):
    """Returns (frame_indices, values). With subsample > 1, libvmaf scores every subsample-th frame only.
    ffmpeg runs inside `tmp` with a RELATIVE log path: an absolute Windows path (C:\\...) would break
    the filter-graph syntax because of the drive colon."""
    lim = ["-t", str(max_seconds)] if max_seconds else []
    rendition, src = os.path.abspath(rendition), os.path.abspath(src)
    if metric == "vmaf":
        graph = (f"[0:v]scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS[d];"
                 f"[1:v]scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS[r];"
                 f"[d][r]libvmaf=log_fmt=json:log_path=vmaf.json:n_threads={max(threads, 1)}:n_subsample={max(subsample, 1)}")
        run(["ffmpeg", "-y", "-i", rendition] + lim + ["-i", src, "-lavfi", graph, "-f", "null", "-"], cwd=tmp)
        with open(os.path.join(tmp, "vmaf.json")) as f:
            frames = json.load(f)["frames"]
        idx = np.array([fr.get("frameNum", i * max(subsample, 1)) for i, fr in enumerate(frames)], np.int64)
        return idx, np.array([fr["metrics"]["vmaf"] for fr in frames], np.float64)
    # PSNR fallback (NOT perceptual; for pipeline testing only)
    graph = (f"[0:v]scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS[d];"
             f"[1:v]scale={w}:{h}:flags=bicubic,setpts=PTS-STARTPTS[r];[d][r]psnr=stats_file=psnr.log")
    run(["ffmpeg", "-y", "-threads", str(threads), "-i", rendition] + lim + ["-i", src, "-lavfi", graph,
         "-f", "null", "-"], cwd=tmp)
    vals = []
    with open(os.path.join(tmp, "psnr.log")) as f:
        for line in f:
            kv = dict(x.split(":") for x in line.split() if ":" in x)
            v = float(kv.get("psnr_avg", "inf")) if kv.get("psnr_avg", "inf") != "inf" else 60.0
            vals.append(v)
    # map PSNR (20..50 dB) to a 0..100 scale so the rest of the pipeline is unchanged
    return np.arange(len(vals)), np.clip((np.array(vals) - 20.0) / 30.0 * 100.0, 0, 100)


def decode_frames(path, size=128, fps=None):
    vf = f"scale={size}:{size}:flags=area" + (f",fps={fps}" if fps else "")
    raw = run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
              capture=True)
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def decode_native(path):
    """Frames at the file's own resolution (RGB, uint8); None if probing fails."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
        w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
        raw = run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture=True)
        return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    except Exception:
        return None


def packet_stats(path, fps, chunk_len, n_chunks):
    out = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=size,flags,pts_time", "-of", "json", path], capture=True))
    feats = np.zeros((n_chunks, 4), np.float32)
    per = [[] for _ in range(n_chunks)]
    for p in out["packets"]:
        k = min(int(float(p.get("pts_time", 0)) // chunk_len), n_chunks - 1)
        per[k].append((int(p["size"]), "K" in p.get("flags", "")))
    for k, lst in enumerate(per):
        if not lst:
            continue
        s = np.array([x[0] for x in lst], np.float64) * 8 / 1e3  # kbit
        key = np.array([x[1] for x in lst])
        feats[k] = [np.log1p(s.mean()), np.log1p(s.std()), np.log1p(s[key].mean() if key.any() else 0),
                    np.log1p(s[~key].mean() if (~key).any() else 0)]
    return feats


def _ascii_path(path):
    """OpenCV on Windows cannot open non-ASCII paths: copy the file to an ASCII temp path."""
    try:
        path.encode("ascii"); return path
    except UnicodeEncodeError:
        import shutil, tempfile
        dst = os.path.join(tempfile.gettempdir(), "nesystream_" + os.path.basename(path))
        try:
            dst.encode("ascii")
        except UnicodeEncodeError:
            dst = os.path.join(os.environ.get("SystemDrive", "C:") + os.sep, "nesystream_tmp", os.path.basename(path))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(path):
            shutil.copy(path, dst)
        return dst


def _init_face_detector():
    """YuNet CNN face detector (cv2.FaceDetectorYN, in core OpenCV >= 4.5.4 incl. 5.x) with the model bundled in
    third_party/opencv; falls back to the Haar cascade (needs opencv-contrib-python on OpenCV 5).
    The model is passed to OpenCV from memory, so non-ASCII folder names on Windows are not a problem."""
    import cv2
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "opencv")
    yunet = os.path.join(here, "face_detection_yunet_2023mar.onnx")
    if hasattr(cv2, "FaceDetectorYN") and os.path.exists(yunet):
        errs = []
        try:
            buf = np.frombuffer(open(yunet, "rb").read(), np.uint8)
            return "yunet", cv2.FaceDetectorYN.create("onnx", buf, np.array([], np.uint8), (320, 320), 0.7, 0.3, 50)
        except Exception as e:
            errs.append(repr(e))
        try:
            return "yunet", cv2.FaceDetectorYN.create(_ascii_path(yunet), "", (320, 320), 0.7, 0.3, 50)
        except Exception as e:
            errs.append(repr(e))
        print("[warn] YuNet face detector failed to load:", " | ".join(errs))
    elif not os.path.exists(yunet):
        print(f"[warn] YuNet model not found at {yunet}")
    else:
        print(f"[warn] this OpenCV ({cv2.__version__}) has no FaceDetectorYN")
    if hasattr(cv2, "CascadeClassifier"):
        for c in [os.path.join(here, "haarcascade_frontalface_default.xml")] + \
                 ([os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")] if hasattr(cv2, "data") else []):
            try:
                if os.path.exists(c):
                    det = cv2.CascadeClassifier(_ascii_path(c))
                    if not det.empty():
                        return "haar", det
            except Exception:
                pass
    return None, None


def _face_count(frame_rgb):
    try:
        import cv2  # optional
    except Exception:
        return 0.0
    if not hasattr(_face_count, "det"):
        _face_count.kind, _face_count.det = _init_face_detector()
        if _face_count.det is None:
            print("[warn] no face detector available: Salient concept will be 0 "
                  "(needs OpenCV with FaceDetectorYN, or opencv-contrib-python for the Haar cascade)")
    if _face_count.det is None:
        return 0.0
    bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
    if _face_count.kind == "yunet":
        h, w = bgr.shape[:2]
        _face_count.det.setInputSize((w, h))
        _, faces = _face_count.det.detect(bgr)
        return 0.0 if faces is None else float(len(faces))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(len(_face_count.det.detectMultiScale(g, 1.1, 4, minSize=(24, 24))))


def visual_features(lowest, fps, chunk_len, n_chunks, n_keyframes=4, size=128):
    """Keyframes and raw concept features per chunk from the lowest rendition."""
    frames = decode_frames(lowest, size)
    fpc = max(int(round(fps * chunk_len)), 1)
    # faces are searched at the lowest rendition's own resolution (e.g. 240p), not in 128x128 thumbnails
    face_frames = decode_native(lowest)
    out_frames = np.zeros((n_chunks, n_keyframes, 3, size, size), np.uint8)
    raw = np.zeros((n_chunks, len(CONCEPT_FEATS)), np.float32)
    gray_all = frames.astype(np.float32).mean(-1) / 255.0
    for k in range(n_chunks):
        seg = frames[k * fpc:(k + 1) * fpc]
        g = gray_all[k * fpc:(k + 1) * fpc]
        if len(seg) == 0:
            seg, g = frames[-1:], gray_all[-1:]
        idx = np.linspace(0, len(seg) - 1, n_keyframes).round().astype(int)
        out_frames[k] = seg[idx].transpose(0, 3, 1, 2)
        gx, gy = np.gradient(g, axis=(1, 2))
        si = float(np.sqrt(gx ** 2 + gy ** 2).std(axis=(1, 2)).max())
        ti = float(np.abs(np.diff(g, axis=0)).mean()) if len(g) > 1 else 0.0
        # scene change: max histogram distance between consecutive frames (incl. boundary to previous chunk)
        gg = gray_all[max(k * fpc - 1, 0):(k + 1) * fpc]
        hists = np.stack([np.histogram(x, 32, (0, 1))[0] / x.size for x in gg])
        scene = float(np.abs(np.diff(hists, axis=0)).sum(1).max()) if len(hists) > 1 else 0.0
        fseg = face_frames[k * fpc:(k + 1) * fpc] if face_frames is not None else None
        face = max(_face_count(fseg[min(i, len(fseg) - 1)]) for i in idx) if fseg is not None and len(fseg) else 0.0
        raw[k] = [ti, si, scene, face, float(g.mean())]
    return out_frames, raw


def process_video(src, out_npz, ladder, chunk_len=4.0, codec="libx264", metric=None, tmp_root=None,
                  max_seconds=None, n_keyframes=4, frame_size=128, keep_renditions=False, threads=0,
                  vmaf_subsample=1):
    """threads: ffmpeg/libvmaf threads for THIS video (0 = all); set it to cores / jobs when running in parallel.
    vmaf_subsample: score every k-th frame (chunk means are barely affected; k=3 is ~3x faster)."""
    metric = metric or ("vmaf" if has_libvmaf() else "psnr")
    if metric == "psnr":
        print("[warn] ffmpeg has no libvmaf: using PSNR-mapped quality (pipeline test only, not for the paper)")
    tmp = tempfile.mkdtemp(dir=tmp_root)
    try:
        w, h, fps, dur = probe(src)
        rend = encode_ladder(src, os.path.join(tmp, "ladder"), ladder, chunk_len, codec, max_seconds=max_seconds,
                             threads=threads)
        sizes = [chunk_sizes(r, chunk_len, tmp) for r in rend]
        n = min(len(s) for s in sizes)
        eff = min(dur, max_seconds) if max_seconds else dur
        n = min(n, int(eff // chunk_len + 1e-6))   # drop a trailing partial chunk
        if n < 1:
            raise ValueError(f"{src} is shorter than one chunk ({eff:.2f}s < {chunk_len}s)")
        fpc = max(int(round(fps * chunk_len)), 1)
        q = []
        for r in rend:
            idx, val = per_frame_quality(r, src, w, h, tmp, metric, max_seconds, threads or os.cpu_count(),
                                         vmaf_subsample)
            ck = idx // fpc
            q.append([val[ck == k].mean() if (ck == k).any() else val[ck <= k][-1] for k in range(n)])
        q = np.array(q, np.float32).T                                    # (N, L)
        q = np.maximum.accumulate(q, axis=1)                             # enforce monotone ladder
        frames, raw = visual_features(rend[0], fps, chunk_len, n, n_keyframes, frame_size)
        cmp_feats = packet_stats(rend[0], fps, chunk_len, n)
        np.savez_compressed(out_npz, sizes=np.stack(sizes, 1)[:n].astype(np.float32), vmaf=q,
                            bitrates_kbps=np.array([b for b, _ in ladder], np.float32),
                            heights=np.array([hh for _, hh in ladder], np.int32), chunk_len=np.float32(chunk_len),
                            frames=frames, cmp_feats=cmp_feats, raw_concepts=raw,
                            metric=np.array(metric), vmaf_subsample=np.int32(vmaf_subsample))
        if keep_renditions:
            shutil.copytree(os.path.join(tmp, "ladder"), os.path.splitext(out_npz)[0] + "_ladder",
                            dirs_exist_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def finalize_concepts(npz_paths, train_paths=None, q=0.75, temp=0.1):
    """Turn raw features into soft concept labels with thresholds fitted on the TRAIN split only.
    HighMotion<-TI, HighTexture<-SI, SceneCut<-hist distance, Salient<-faces, Dark<-luma."""
    train_paths = train_paths or npz_paths
    raw_tr = np.concatenate([np.load(p)["raw_concepts"] for p in train_paths])
    thr = np.quantile(raw_tr, q, axis=0)
    scale = raw_tr.std(0) + 1e-6
    for p in npz_paths:
        d = dict(np.load(p, allow_pickle=False))
        r = d["raw_concepts"]
        c = np.zeros_like(r)
        c[:, 0] = 1 / (1 + np.exp(-(r[:, 0] - thr[0]) / (temp * scale[0])))
        c[:, 1] = 1 / (1 + np.exp(-(r[:, 1] - thr[1]) / (temp * scale[1])))
        c[:, 2] = 1 / (1 + np.exp(-(r[:, 2] - 0.5) / 0.05))            # absolute histogram-distance threshold
        c[:, 3] = np.clip(r[:, 3], 0, 1)                               # any face in a keyframe
        c[:, 4] = 1 / (1 + np.exp(-(0.25 - r[:, 4]) / 0.03))           # mean luma < 0.25
        d["concepts"] = c.astype(np.float32)
        np.savez_compressed(p, **d)
    return thr
