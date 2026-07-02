#!/usr/bin/env python3
"""
P2 demo: adaptive runtime index update under distribution drift (paper sec IV-B-3).

Loads VLiteEngine with a DELIBERATELY STALE partition (hot clusters that cover only
~8% of the trace's accesses, simulating a query distribution that has drifted away
from what the index was partitioned for). Streams the real Search-R1 trace queries
through it with --enable_swap, a small refresh window, and an SLO threshold. Shows:

  phase 1 (stale):  low hit-rate -> most probes hit the CPU cold tier -> high
                    retrieval latency -> SLO attainment below threshold
  -> REPARTITION fires in the background (rebuilds hot shards from live access)
  phase 2 (adapted): hot set realigned -> latency drops -> SLO recovers

Run:  source tools/sr1/env.sh && $VLITE_PY tools/sr1/drift_demo.py
"""
import os
import re
import sys
import json
import time
import argparse
import numpy as np

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)


def log(m):
    print(f"[DRIFT-DEMO] {m}", flush=True)


def load_queries(trace, n):
    qs = []
    with open(trace) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("question"):
                qs.append(r["question"])
            for t in r.get("trace_turns", []) or []:
                for m in SEARCH_RE.findall(t.get("llm_output", "") or ""):
                    if m.strip():
                        qs.append(m.strip())
            if len(qs) >= n:
                break
    return qs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part_dir", default="database/sr1/4gpus_stale")
    ap.add_argument("--nprobe", type=int, default=256)
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--search_slo_ms", type=float, default=400.0)
    ap.add_argument("--refresh_interval", type=int, default=200)
    ap.add_argument("--slo_threshold", type=float, default=0.9)
    ap.add_argument("--n_queries", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from vliterag.vlite_engine import VLiteEngine
    eng = VLiteEngine(os.environ["SR1_INDEX"], args.part_dir, os.environ["SR1_E5"],
                      nprobe=args.nprobe, topk=3, ngpu=args.ngpu,
                      search_slo_ms=args.search_slo_ms, enable_swap=True,
                      refresh_interval=args.refresh_interval,
                      slo_threshold=args.slo_threshold)

    qs = load_queries(os.environ["SR1_TRACE"], args.n_queries)
    log(f"loaded {len(qs)} queries; SLO={args.search_slo_ms}ms "
        f"refresh={args.refresh_interval} threshold={args.slo_threshold}")
    qv = eng.encode(qs)

    # stream in fixed batches; record per-window mean retrieval latency + SLO
    win, win_lat, win_ok, win_n = [], 0.0, 0, 0
    window_id = 0
    hist = []
    for i in range(0, len(qv), args.batch):
        sub = qv[i:i + args.batch]
        t = time.time()
        eng.search_vectors(sub, 3)
        lat = (time.time() - t) * 1000
        per_q = lat / sub.shape[0]
        win_lat += lat
        win_ok += sub.shape[0] if per_q <= args.search_slo_ms else 0
        win_n += sub.shape[0]
        if win_n >= args.refresh_interval:
            scan = eng.last_scan_ms
            attain = win_ok / win_n
            hot = int((eng.shard_lut != -1).sum())
            log(f"window {window_id}: mean_scan={scan:.0f}ms slo_attain={attain:.2f} "
                f"hot_clusters={hot}")
            hist.append((window_id, scan, attain, hot))
            window_id += 1
            win_lat, win_ok, win_n = 0.0, 0, 0

    # wait for any in-flight repartition to settle, then one more pass
    if eng._repart_thread is not None:
        log("waiting for in-flight repartition to finish...")
        eng._repart_thread.join(timeout=180)
    log("=== post-adaptation check ===")
    t = time.time()
    for i in range(0, min(len(qv), args.refresh_interval * args.batch // args.batch), args.batch):
        eng.search_vectors(qv[i:i + args.batch], 3)
    log(f"final scan_ms sample = {eng.last_scan_ms:.0f}ms  hot_clusters={int((eng.shard_lut!=-1).sum())}")

    log("=== summary (window, scan_ms, slo_attain, hot_clusters) ===")
    for w in hist:
        log(f"  {w}")


if __name__ == "__main__":
    main()
