#!/usr/bin/env python3
"""
Collect Search-R1 replay points in ISOLATION: one fresh vLLM process per rps so
points cannot interfere (no KV-cache/queue/GPU-fragmentation carryover). Picks the
freest GPU per point, waits for it to actually be free, runs replay.py for a single
rps, validates the result, and re-runs on failure or anomaly. Per-point output goes
to runs/vlite_full/rps<X>/.

Usage:  source tools/sr1/env.sh
        $VLITE_PY tools/sr1/collect_points.py --rps -1,0.5,0.75,1.25,1.5,2.5,3.5,4,4.5,5
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def log(m):
    print(f"[COLLECT] {m}", flush=True)


def gpu_free_mib():
    """Return list of (index, free_MiB) sorted by most-free first."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    ).decode()
    rows = []
    for line in out.strip().splitlines():
        idx, free = line.split(",")
        rows.append((int(idx), int(free)))
    return sorted(rows, key=lambda r: -r[1])


def wait_for_gpu(min_free_mib=19000, timeout_s=600):
    """Block until some GPU has >= min_free_mib free; return its index."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        rows = gpu_free_mib()
        idx, free = rows[0]
        if free >= min_free_mib:
            return idx
        log(f"waiting for a free GPU (best: GPU{idx} {free} MiB free)...")
        time.sleep(15)
    raise RuntimeError("no GPU became free in time")


def is_anomaly(rps, summary, log_text):
    """Return (is_bad, reason). Conservative: only clear failures / stalls."""
    if summary is None:
        return True, "no summary.json"
    ss = summary.get("summaries", [])
    if not ss:
        return True, "empty summaries"
    s = ss[0]
    if s.get("n", 0) < summary.get("_limit", 0):
        return True, f"only {s.get('n')} records completed"
    # fatal runtime failures (NOTE: a bare 'Traceback' is NOT fatal — vLLM's metrics
    # logger emits a benign `AttributeError: module 'tabulate'...` traceback that
    # does not affect the replay; only flag the real engine/GPU failures).
    for bad in ["out of memory", "OutOfMemoryError", "EngineDeadError",
                "Faiss assertion", "CUDA error", "less than desired GPU"]:
        if bad in log_text:
            return True, f"log contains '{bad}'"
    # retrieval compute must be in a sane band (real IVF-Flat hybrid ~0.2-3s)
    rc = s.get("retr_compute_avg", 0.0)
    if rc <= 0 or rc > 10000:
        return True, f"retr_compute_avg out of band ({rc:.0f}ms)"
    # low-rps stall: at light load throughput should track the offered rps
    if 0 < rps <= 1.5 and s.get("e2e_throughput", 0) < 0.6 * rps:
        return True, f"throughput stall ({s['e2e_throughput']:.2f} << {rps})"
    return False, "ok"


def run_point(rps, args, attempt):
    tag = f"rps{rps}"
    out_dir = ROOT / args.out_root / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu = wait_for_gpu(args.min_free_mib)
    log(f"{tag} attempt {attempt}: using GPU{gpu}")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [
        args.py, str(ROOT / "tools/sr1/replay.py"),
        "--retriever", "vlite", "--part_dir", args.part_dir,
        "--rps_list", str(rps), "--limit", str(args.limit),
        "--nprobe", str(args.nprobe), "--topk", str(args.topk),
        "--ngpu", "1", "--tp", "1", "--gpu_mem_util", str(args.gpu_mem_util),
        "--search_slo_ms", str(args.search_slo_ms), "--seed", str(args.seed),
        "--max_model_len", str(args.max_model_len),
        "--out_dir", str(out_dir),
    ]
    if args.dispatcher:
        cmd += ["--dispatcher", "--disp_max_batch", str(args.disp_max_batch),
                "--disp_window_ms", str(args.disp_window_ms)]
    cmd += ["--no_thinking"] if args.no_thinking else ["--thinking"]

    logf = out_dir / "driver.log"
    with open(logf, "w") as lf:
        rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode

    log_text = logf.read_text(errors="ignore")
    summ = None
    sp = out_dir / "summary.json"
    if sp.exists():
        summ = json.load(open(sp))
        summ["_limit"] = args.limit
    bad, reason = is_anomaly(rps, summ, log_text)
    # exit code is advisory only: vLLM's atexit/logging can raise a benign error
    # after the replay already wrote a complete summary.json. Trust summary
    # completeness (checked in is_anomaly) as the ground-truth success signal.
    if rc != 0 and bad:
        reason = f"exit {rc}; {reason}"
    return (not bad), reason, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rps", required=True, help="comma-separated rps points")
    ap.add_argument("--out_root", default="runs/vlite_full")
    ap.add_argument("--part_dir", default="database/sr1/1gpu_3g")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--nprobe", type=int, default=512)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--gpu_mem_util", type=float, default=0.6)
    ap.add_argument("--search_slo_ms", type=float, default=400.0)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--dispatcher", action="store_true", default=True)
    ap.add_argument("--disp_max_batch", type=int, default=16)
    ap.add_argument("--disp_window_ms", type=float, default=5.0)
    ap.add_argument("--no_thinking", action="store_true", default=True)
    ap.add_argument("--min_free_mib", type=int, default=19000)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--py", default=os.environ.get("VLITE_PY", sys.executable))
    args = ap.parse_args()

    points = [float(x) for x in args.rps.split(",")]
    results = {}
    for rps in points:
        for attempt in range(1, args.max_retries + 2):
            ok, reason, summ = run_point(rps, args, attempt)
            if ok:
                s = summ["summaries"][0]
                log(f"rps={rps} OK: e2e_tput={s['e2e_throughput']:.2f} "
                    f"e2e_p50/p90/p99={s['e2e_p50']:.0f}/{s['e2e_p90']:.0f}/{s['e2e_p99']:.0f} "
                    f"kv_hit={s['kv_cache_hit_rate']:.3f} retr_comp={s['retr_compute_avg']:.0f} "
                    f"retr_queue={s['retr_queue_avg']:.0f}")
                results[rps] = s
                break
            log(f"rps={rps} attempt {attempt} ANOMALY/FAIL: {reason} -> retrying")
            time.sleep(10)
        else:
            log(f"rps={rps} FAILED after {args.max_retries+1} attempts: {reason}")
            results[rps] = {"failed": reason}
    log(f"collected {len([r for r in results.values() if 'failed' not in r])}/{len(points)} points")


if __name__ == "__main__":
    main()
