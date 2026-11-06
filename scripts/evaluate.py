"""End-to-end evaluation (E3/E5): all test videos x test traces, parallel over CPU processes.

python scripts/evaluate.py --video_dir data/videos --trace_dir data/traces/test/hsdpa --policy runs/policy --out runs/eval.json
  --jobs 16          worker processes (default: physical-core estimate); 1 = in-process
  --max_traces 20    quick pilot
Outputs: <out> (mean per method) and <out>_sessions.json (one row per session: video, trace, source, metrics),
for significance tests and film / short-clip subsets. Every session uses a fixed seed -> reproducible.
"""
import _common  # noqa
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np, torch


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True); ap.add_argument("--trace_dir", required=True)
    ap.add_argument("--policy", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--hard_tput", default="", help="inference-time override of H1's throughput estimate: low | hat")
    ap.add_argument("--hard_off_buffer", type=float, default=-1, help="inference-time override: skip H1 above this buffer (s)")
    ap.add_argument("--b_min", type=float, default=-1, help="inference-time override of H1's safety margin (s)")
    ap.add_argument("--trace_sample", type=int, default=0, help="random subset of N test traces (fixed seed)")
    ap.add_argument("--max_traces", type=int, default=None); ap.add_argument("--teacher_steps", default="20,4")
    ap.add_argument("--student_steps", default="1,2,4")
    ap.add_argument("--baselines_dir", default=None, help="PPO-family checkpoints from train_baselines.py")
    ap.add_argument("--diffbase_dir", default=None, help="GenPO / VDRL checkpoints (*.ptd) from train_diffusion_baselines.py")
    ap.add_argument("--newbase_dir", default=None, help="EAStream / NeuroBA checkpoints (*.ptn) from train_new_baselines.py")
    ap.add_argument("--extra_policies", default="", help="name=dir,... diffusion baselines (DIPO/QSM/ablations)")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--block", type=int, default=10, help="sessions per task")
    ap.add_argument("--only", default="", help="comma list of method names to evaluate")
    ap.add_argument("--hys", default="", help="comma list of switch-hysteresis weights for critic selection "
                    "(score = Q(l) - lam*|l - prev|, previous level added as a candidate); student 1-step only")
    ap.add_argument("--val_bw_aug", default="", help="validation only: rescale TRAINING traces (log-uniform lo,hi Mbps)")
    ap.add_argument("--plan", default="", help="comma list of look-ahead horizons H for the rule-based planner "
                    "(Ours-plan<H>-best<K>, -all, -q)")
    ap.add_argument("--plan_lams", default="", help="comma list of data costs (reward units per MB) -> Ours-plan<H>-all-lam<x>")
    ap.add_argument("--plan_wq", type=float, default=1.0, help="weight of Q(s,a) in Ours-plan<H>-q (reward units)")
    ap.add_argument("--conf_file", default="", help="with --hard_tput conformal: calibration ratios")
    ap.add_argument("--conf_alpha_hi", type=float, default=0.0, help="buffer-adaptive alpha at a full buffer (0 = off)")
    ap.add_argument("--conf_buf", default="10,30", help="buffer range (s) over which alpha rises to --conf_alpha_hi")
    ap.add_argument("--plan_obj_qs", default="", help="comma list of quantiles q: planner objective uses c_hat x Q_q(ratios) "
                    "(H1 feasibility keeps c_h1) -> Ours-plan<H>-all-oq<q>")
    ap.add_argument("--conf_alpha", type=float, default=0.1); ap.add_argument("--conf_gamma", type=float, default=0.01)
    ap.add_argument("--dump_pairs", default="", help="video@trace;... : evaluate only these sessions and save per-chunk logs")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"], help="video split to evaluate on")
    ap.add_argument("--ens", default="", help="comma list of extra policy dirs (other seeds): their students propose "
                    "candidates and all critics are averaged (Ours-ens<N>-best<K>[-hys<lam>])")
    ap.add_argument("--extra_k", default="", help="comma list of additional K values for the 1-step student")
    ap.add_argument("--best_k", type=int, default=8,
                    help="test-time action selection: sample K candidates, keep the one with the highest critic "
                         "value (methods with suffix -best<K>); 0 = off")
    return ap.parse_args(argv)


def build(a):
    """Load data and construct every policy (called once per worker process)."""
    from nesystream.env.abr_env import ABREnv, EnvConfig, load_traces, load_video
    from nesystream.data.chunks import list_videos, split_by_video, source_of
    from nesystream.models.nsqoe import NSQoE
    from nesystream.models.diffusion_policy import DiffusionPolicy, ConsistencyStudent
    from nesystream.eval.baselines import BBA, RB, RobustMPC
    _tr, _va, te = split_by_video(list_videos(a.video_dir), os.path.join(a.video_dir, "split.json"))
    te = {"test": te, "val": _va, "train": _tr}[a.split]       # val/train: for tuning test-time options only
    videos = [load_video(p) for p in te]
    traces = load_traces(a.trace_dir, a.max_traces)
    if a.trace_sample and a.trace_sample < len(traces):   # random subset, seed 0, kept in file order
        idx = sorted(np.random.default_rng(0).choice(len(traces), a.trace_sample, replace=False))
        traces = [traces[i] for i in idx]
    ck = torch.load(os.path.join(a.policy, "teacher.pt"), map_location="cpu")
    ecfg = EnvConfig(**ck["env"]); ecfg.bw_aug = a.val_bw_aug   # never rescale the TEST traces (default '')
    if a.hard_tput: ecfg.hard_tput = a.hard_tput
    if a.conf_file: ecfg.conf_file, ecfg.conf_alpha, ecfg.conf_gamma = a.conf_file, a.conf_alpha, a.conf_gamma
    if a.conf_alpha_hi > 0:
        ecfg.conf_alpha_hi = a.conf_alpha_hi; ecfg.conf_buf_lo, ecfg.conf_buf_hi = (float(x) for x in a.conf_buf.split(","))
    if a.hard_off_buffer >= 0: ecfg.hard_off_buffer = a.hard_off_buffer
    if a.b_min >= 0: ecfg.b_min = a.b_min
    env = ABREnv(videos, traces, NSQoE(init_weights=ck["nsqoe"]), ecfg)
    pi = DiffusionPolicy(ck["obs_dim"], ck["L"], ck["cfg"]["horizon"], ck["cfg"]["T"]); pi.load_state_dict(ck["pi"]); pi.eval()
    eta = ck["cfg"]["eta_guid"]

    def ctx(e, o):
        return (torch.as_tensor(o)[None], torch.as_tensor(e.ctx["energy"])[None], torch.tensor([e.ctx["l_max"]]))

    def teacher(n, guided=True):
        def f(e, o):
            ob, en, lm = ctx(e, o)
            with torch.no_grad():
                return int(pi.sample(ob, n, en if guided else None, eta if guided else 0.0, lm)[0])
        return f

    pols = {"BBA": lambda e, o: BBA().act(e), "RB": lambda e, o: RB().act(e),
            "RobustMPC-bitrate": lambda e, o: RobustMPC(5, "bitrate").act(e),
            "RobustMPC-quality": lambda e, o: RobustMPC(5, "quality").act(e)}
    for n in [int(x) for x in a.teacher_steps.split(",")]:
        pols[f"Ours-teacher-{n}step"] = teacher(n)
    pols["Ours-noguidance"] = teacher(int(a.teacher_steps.split(",")[0]), guided=False)
    # ---- test-time selection with the critic: best-of-K samples, and critic-only greedy (ablation)
    from nesystream.models.diffusion_policy import QEnsemble
    Q = None
    if "q" in ck:
        Q = QEnsemble(ck["obs_dim"], ck["L"], ck["cfg"]["n_q"]); Q.load_state_dict(ck["q"]); Q.eval()

    def best_of(sampler, K, lam=0.0):
        def f(e, o):
            ob, en, lm = ctx(e, o)
            with torch.no_grad():
                lv = sampler(ob.repeat(K, 1), en.repeat(K, 1), lm.repeat(K)).view(-1)
                qv = Q(ob).mean(0)[0]                       # (L,) critic value of every level
            if lam > 0 and e.log:
                prev = int(e.log[-1]["level"])
                if prev <= int(lm[0]):                      # keeping the level is a candidate if it is feasible
                    lv = torch.cat([lv, torch.tensor([prev], dtype=lv.dtype)])
                sc = qv[lv] - lam * (lv - prev).abs().float()
                return int(lv[torch.argmax(sc)])
            return int(lv[torch.argmax(qv[lv])])
        return f

    def plan_scores(e, roots, H, lam=0.0, obj_q=0.0):
        """Rule-based look-ahead: for every root level, the best H-step NS-QoE return over branches that stay,
        go one level up or one level down, all inside H1 (robust throughput c_low). Returns (n_roots,) in reward units."""
        L, N, k = e.L, e.N, e.k
        c_feas = torch.tensor([float(e.ctx.get("c_h1", e.ctx["c_low"]))])
        if obj_q > 0:      # objective / buffer simulation with a central throughput estimate, feasibility with c_h1
            if not hasattr(e, "_conf_r"):
                e._c_conformal(1.0)
            c = torch.tensor([float(e.ctx["c_hat"]) * float(np.quantile(e._conf_r, obj_q))])
        else:
            c = c_feas
        rtt, cl = e.cfg.rtt, e.chunk_len
        lv = torch.as_tensor(roots, dtype=torch.long); root = torch.arange(len(lv))
        buf = torch.full((len(lv),), float(e.buffer)); ret = torch.zeros(len(lv))
        prev_q = torch.full((len(lv),), float(e.prev_q if e.prev_q is not None else e.ctx["q_curve"][0]))
        for j in range(min(H, N - k)):
            kk = k + j
            qh = torch.as_tensor(e._quality_curve(kk), dtype=torch.float32)
            sz = torch.as_tensor(e.v["sizes"][kk], dtype=torch.float32)
            cc = torch.as_tensor(e._concepts(kk), dtype=torch.float32)
            if j > 0:     # expand: stay / +1 / -1, keep H1-feasible children (level 0 always allowed)
                ch = (lv[:, None] + torch.tensor([0, 1, -1])[None]).clamp(0, L - 1)
                n = len(lv)
                lv, root, buf, ret, prev_q = (ch.reshape(-1), root.repeat_interleave(3), buf.repeat_interleave(3),
                                              ret.repeat_interleave(3), prev_q.repeat_interleave(3))
                dl = sz[lv] * 8.0 / c_feas + rtt
                ok = ((buf - dl) >= e.cfg.b_min) | (lv == 0)
                lv, root, buf, ret, prev_q = lv[ok], root[ok], buf[ok], ret[ok], prev_q[ok]
            B = len(lv)
            with torch.no_grad():
                tab = e.q.step_qoe_all_levels(qh.expand(B, L), sz.expand(B, L), c.expand(B), buf, prev_q,
                                              cc.expand(B, -1), torch.full((B,), kk / N), rtt)
            ret = ret + (0.99 ** j) * (tab.gather(1, lv[:, None]).squeeze(1) / 100.0 - lam * sz[lv] / 1e6)
            dl = sz[lv] * 8.0 / c + rtt
            buf = (buf - dl).clamp_min(0) + cl
            prev_q = qh[lv]
        best = torch.full((len(roots),), -1e9)
        return best.scatter_reduce(0, root, ret, reduce="amax")

    def planner(sampler, K, H, roots="samples", wq=0.0, lam=0.0, obj_q=0.0):
        def f(e, o):
            ob, en, lm = ctx(e, o)
            if roots == "all":
                cand = torch.arange(int(lm[0]) + 1)
            else:
                with torch.no_grad():
                    cand = torch.unique(sampler(ob.repeat(K, 1), en.repeat(K, 1), lm.repeat(K)).view(-1))
            sc = plan_scores(e, cand.tolist(), H, lam, obj_q)
            if wq > 0:
                with torch.no_grad():
                    sc = sc + wq * Q(ob).mean(0)[0][cand]
            return int(cand[torch.argmax(sc)])
        return f

    def q_greedy(e, o):
        ob, en, lm = ctx(e, o)
        with torch.no_grad():
            qv = Q(ob).mean(0)[0].clone()
        qv[int(lm[0]) + 1:] = -float("inf")                # stay inside the feasible set
        return int(torch.argmax(qv))
    K = a.best_k
    if Q is not None and K > 0:
        for n in [int(x) for x in a.teacher_steps.split(",")]:
            pols[f"Ours-teacher-{n}step-best{K}"] = best_of(
                (lambda n: lambda ob, en, lm: pi.sample(ob, n, en, eta, lm))(n), K)
        pols["Ours-Qgreedy"] = q_greedy
        n0 = int(a.teacher_steps.split(",")[0])     # fair guidance ablation: same best-of-K, energy guidance off
        pols[f"Ours-noguidance-best{K}"] = best_of(lambda ob, en, lm: pi.sample(ob, n0, None, 0.0, lm), K)
    sp = os.path.join(a.policy, "student.pt")
    if os.path.exists(sp):
        st = ConsistencyStudent(ck["obs_dim"], ck["L"], ck["cfg"]["horizon"], pi.alpha_bar)
        st.load_state_dict(torch.load(sp, map_location="cpu")["student"]); st.eval()
        for n in [int(x) for x in a.student_steps.split(",")]:
            pols[f"Ours-student-{n}step"] = (lambda n: lambda e, o: int(st.sample(ctx(e, o)[0], n_steps=n, l_max=ctx(e, o)[2])[0]))(n)
            if Q is not None and K > 0:
                pols[f"Ours-student-{n}step-best{K}"] = best_of(
                    (lambda n: lambda ob, en, lm: st.sample(ob, n_steps=n, l_max=lm))(n), K)
                if n == 1:
                    s1 = lambda ob, en, lm: st.sample(ob, n_steps=1, l_max=lm)
                    for k2 in [int(x) for x in a.extra_k.split(",") if x]:
                        pols[f"Ours-student-1step-best{k2}"] = best_of(s1, k2)
                    for lam in [float(x) for x in a.hys.split(",") if x]:
                        pols[f"Ours-student-1step-best{K}-hys{lam:g}"] = best_of(s1, K, lam)
                    for H in [int(x) for x in a.plan.split(",") if x]:
                        pols[f"Ours-plan{H}-best{K}"] = planner(s1, K, H, roots="samples")
                        pols[f"Ours-plan{H}-all"] = planner(s1, K, H, roots="all")
                        pols[f"Ours-plan{H}-q"] = planner(s1, K, H, roots="samples", wq=a.plan_wq)
                        for oq in [float(x) for x in a.plan_obj_qs.split(",") if x]:
                            pols[f"Ours-plan{H}-all-oq{oq:g}"] = planner(s1, K, H, roots="all", obj_q=oq)
                        for lam in [float(x) for x in a.plan_lams.split(",") if x]:
                            pols[f"Ours-plan{H}-all-lam{lam:g}"] = planner(s1, K, H, roots="all", lam=lam)
                    if a.ens:
                        sts, Qs = [st], [Q]
                        for d in a.ens.split(","):
                            c3 = torch.load(os.path.join(d, "teacher.pt"), map_location="cpu")
                            q3 = QEnsemble(c3["obs_dim"], c3["L"], c3["cfg"]["n_q"]); q3.load_state_dict(c3["q"]); q3.eval()
                            s3 = ConsistencyStudent(c3["obs_dim"], c3["L"], c3["cfg"]["horizon"], pi.alpha_bar)
                            s3.load_state_dict(torch.load(os.path.join(d, "student.pt"), map_location="cpu")["student"]); s3.eval()
                            sts.append(s3); Qs.append(q3)

                        def ens_of(lam, sts=sts, Qs=Qs):
                            def f(e, o):
                                ob, en, lm = ctx(e, o)
                                with torch.no_grad():
                                    lv = torch.cat([m.sample(ob.repeat(K, 1), n_steps=1, l_max=lm.repeat(K)).view(-1) for m in sts])
                                    qv = torch.stack([q(ob).mean(0)[0] for q in Qs]).mean(0)
                                if lam > 0 and e.log:
                                    prev = int(e.log[-1]["level"])
                                    if prev <= int(lm[0]):
                                        lv = torch.cat([lv, torch.tensor([prev], dtype=lv.dtype)])
                                    return int(lv[torch.argmax(qv[lv] - lam * (lv - prev).abs().float())])
                                return int(lv[torch.argmax(qv[lv])])
                            return f
                        n_e = len(sts)
                        pols[f"Ours-ens{n_e}-best{K}"] = ens_of(0.0)
                        for lam in [float(x) for x in a.hys.split(",") if x]:
                            pols[f"Ours-ens{n_e}-best{K}-hys{lam:g}"] = ens_of(lam)
    from nesystream.eval.learned_baselines import ActorCritic, ppo_act_fn

    def wrap_src(fn, src, log_tput=None):
        """Run a policy that observes with its OWN training config (quality_source, throughput scale) on the same
        session; the environment dynamics and the scoring are unchanged."""
        over = {"quality_source": src}
        if log_tput is not None:
            over["log_tput"] = bool(log_tput)

        def f(e, o):
            old = {k: getattr(e.cfg, k) for k in over}
            for k, v in over.items():
                setattr(e.cfg, k, v)
            ob = e._obs()
            for k, v in old.items():
                setattr(e.cfg, k, v)
            a_ = fn(e, ob); e._obs()
            return a_
        return f
    if a.baselines_dir:
        import glob
        for p in sorted(glob.glob(os.path.join(a.baselines_dir, "*.pt"))):
            b = torch.load(p, map_location="cpu")
            net = ActorCritic(b["obs_dim"], b["L"], mask_tail=b.get("mask_tail", 0))
            net.load_state_dict(b["state"], strict=False); net.eval()
            pols[os.path.splitext(os.path.basename(p))[0]] = wrap_src(ppo_act_fn(net), b["quality_source"], b.get("log_tput", False))
    if a.newbase_dir:   # 2026 baselines (EAStream, NeuroBA) from train_new_baselines.py
        import glob
        from nesystream.eval.new_baselines import load_new_baseline
        for p in sorted(glob.glob(os.path.join(a.newbase_dir, "*.ptn"))):
            nm, fn, src, lt = load_new_baseline(p)
            pols[nm] = wrap_src(fn, src, lt)
    if a.diffbase_dir:  # NeurIPS 2025 diffusion-RL baselines (GenPO, VDRL) from train_diffusion_baselines.py
        import glob
        from nesystream.eval.diffusion_baselines import load_diffusion_baseline
        for p in sorted(glob.glob(os.path.join(a.diffbase_dir, "*.ptd"))):
            nm, fn, src, lt = load_diffusion_baseline(p)
            pols[nm] = wrap_src(fn, src, lt)
    for item in filter(None, a.extra_policies.split(",")):
        name, d = item.split("=")
        c2 = torch.load(os.path.join(d, "teacher.pt"), map_location="cpu")
        if c2["cfg"].get("policy", "diffusion") == "categorical":
            from nesystream.rl.trainer import CategoricalPolicy
            p2 = CategoricalPolicy(c2["obs_dim"], c2["L"])
        else:
            p2 = DiffusionPolicy(c2["obs_dim"], c2["L"], c2["cfg"]["horizon"], c2["cfg"]["T"])
        p2.load_state_dict(c2["pi"]); p2.eval()

        def mk(p2=p2, c2=c2):
            def f(e, o):
                hard = c2["env"]["use_hard_rules"]; g = c2["cfg"]["eta_guid"] > 0
                ob, en, lm = ctx(e, o)
                with torch.no_grad():
                    return int(p2.sample(ob, c2["cfg"]["T"], en if g else None, c2["cfg"]["eta_guid"], lm if hard else None)[0])
            return f
        pols[name] = wrap_src(mk(), c2["env"]["quality_source"], c2["env"].get("log_tput", False))
        if K > 0 and "q" in c2 and c2["cfg"].get("policy", "diffusion") == "diffusion":
            # same test-time selection as ours: K samples, keep the one its OWN critic values most
            Q2 = QEnsemble(c2["obs_dim"], c2["L"], c2["cfg"]["n_q"]); Q2.load_state_dict(c2["q"]); Q2.eval()

            def mk_best(p2=p2, c2=c2, Q2=Q2):
                hard = c2["env"]["use_hard_rules"]; g = c2["cfg"]["eta_guid"] > 0

                def f(e, o):
                    ob, en, lm = ctx(e, o)
                    with torch.no_grad():
                        lv = p2.sample(ob.repeat(K, 1), c2["cfg"]["T"], en.repeat(K, 1) if g else None,
                                       c2["cfg"]["eta_guid"], lm.repeat(K) if hard else None).view(-1)
                        qv = Q2(ob).mean(0)[0]
                    return int(lv[torch.argmax(qv[lv])])
                return f
            pols[f"{name}-best{K}"] = wrap_src(mk_best(), c2["env"]["quality_source"], c2["env"].get("log_tput", False))
    if a.only:
        keep = set(a.only.split(","))
        pols = {k: v for k, v in pols.items() if k in keep}
    H = videos[0].get("heights")   # per-rung display resolutions for P.1203 (16:9, e.g. 240 -> 426x240)
    resolutions = [f"{int(round(h * 16 / 9 / 2)) * 2}x{int(h)}" for h in H] if H is not None else None
    pairs = [(v, t) for v in range(len(videos)) for t in range(len(traces))]
    if a.dump_pairs:
        vn = [os.path.splitext(os.path.basename(p))[0] for p in te]; tn = [t["name"] for t in traces]
        want = [x.split("@") for x in a.dump_pairs.split(";") if x]
        pairs = [(vn.index(v), tn.index(t)) for v, t in want]
    return {"env": env, "pols": pols, "pairs": pairs, "resolutions": resolutions, "dump": bool(a.dump_pairs),
            "video_names": [os.path.splitext(os.path.basename(p))[0] for p in te],
            "sources": [source_of(p) for p in te], "trace_names": [t["name"] for t in traces]}


_G = {}


def _init(argv):
    torch.set_num_threads(1)            # one core per worker
    import logging; logging.disable(logging.WARNING)
    _G.update(build(parse(argv)))


def _task(job):
    from nesystream.eval.evaluate import run_policy
    name, block = job
    rows = []
    for v, t in block:
        seed = 1000003 * v + t        # session seed: same result in any process / order
        torch.manual_seed(seed); np.random.seed(seed % 2 ** 32)
        _G["env"].rng = np.random.default_rng(seed)    # env noise (throughput jitter) is drawn from env.rng
        r = run_policy(_G["env"], _G["pols"][name], v, t, _G["resolutions"])
        r.update({"video": _G["video_names"][v], "source": _G["sources"][v], "trace": _G["trace_names"][t]})
        if _G.get("dump"):
            e = _G["env"]; src = e.v.get("chunk_src")
            r["log"] = [{kk: (float(vv) if isinstance(vv, (int, float, np.integer, np.floating, bool, np.bool_)) else vv)
                         for kk, vv in x.items()} | ({"src": list(map(str, src[i]))} if src is not None and i < len(src) else {})
                        for i, x in enumerate(e.log)]
            r["trace_mbps"] = [float(x) for x in e.tr["mbps"]]; r["trace_time"] = [float(x) for x in e.tr["time"]]
        rows.append(r)
    return name, rows


def main():
    a = parse()
    assert not (a.val_bw_aug and a.split == "test"), "--val_bw_aug is for validation on the train/val split only"
    from nesystream.eval.evaluate import print_table, save, bandwidth_saving
    _init(sys.argv[1:])
    names, pairs = list(_G["pols"]), _G["pairs"]
    print("P.1203 resolutions:", _G["resolutions"])
    jobs = [(n, pairs[i:i + a.block]) for n in names for i in range(0, len(pairs), a.block)]
    total = len(names) * len(pairs)
    print(f"{len(names)} methods x {len(pairs)} sessions = {total} sessions, {len(jobs)} tasks, {a.jobs} workers", flush=True)
    rows = {n: [] for n in names}
    t0, done, last = time.time(), 0, 0.0

    def collect(name, rs):
        nonlocal done, last
        rows[name] += rs; done += len(rs)
        el = time.time() - t0
        if el - last > 20 or done == total:
            last = el
            print(f"  {done}/{total} sessions  {el / 60:.1f} min elapsed, ~{el / done * (total - done) / 60:.1f} min left", flush=True)
    if a.jobs <= 1:
        for j in jobs:
            collect(*_task(j))
    else:
        with ProcessPoolExecutor(a.jobs, mp_context=mp.get_context("spawn"), initializer=_init,
                                 initargs=(sys.argv[1:],)) as ex:
            for f in as_completed([ex.submit(_task, j) for j in jobs]):
                collect(*f.result())
    metric_keys = [k for k in rows[names[0]][0] if k not in ("video", "source", "trace", "log", "trace_mbps", "trace_time")]
    res = {n: {k: float(np.nanmean([r[k] for r in rows[n]])) for k in metric_keys} for n in names}
    print_table(res)
    ref = "RobustMPC-bitrate"
    if ref in res:
        for k in res:
            res[k]["bytes_saving_vs_" + ref] = bandwidth_saving(res, ref, k)
    save(res, a.out)
    with open(os.path.splitext(a.out)[0] + "_sessions.json", "w") as f:
        json.dump(rows, f)
    print("saved", a.out, "and", os.path.splitext(a.out)[0] + "_sessions.json",
          f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
