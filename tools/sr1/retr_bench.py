#!/usr/bin/env python3
"""Retrieval-only microbench across partitions (rho sweep). Measures pure hybrid
CPU/GPU IVF-Flat search latency per partition at fixed nprobe -> the serialized
retrieval SERVICE CAPACITY (searches/s) that caps single-GPU throughput. No vLLM,
so it isolates the coverage(rho) effect from KV-cache contention.

Usage: source tools/sr1/env.sh
       CUDA_VISIBLE_DEVICES=<free> $VLITE_PY tools/sr1/retr_bench.py \
           --parts database/sr1/1gpu_3g,database/sr1/1gpu_cov35,...  --nprobe 512
"""
import os, sys, json, time, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True, help="comma-separated partition dirs")
    ap.add_argument("--nprobe", type=int, default=512)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--nq", type=int, default=64, help="queries to time")
    ap.add_argument("--batch", type=int, default=1, help="search batch size")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    from vliterag.vlite_engine import VLiteEngine
    import faiss
    faiss.omp_set_num_threads(32)

    # load real trace queries once
    qs = []
    with open(os.environ["SR1_TRACE"]) as f:
        for line in f:
            r = json.loads(line)
            if r.get("question"):
                qs.append(r["question"])
            if len(qs) >= args.nq:
                break

    print(f"{'partition':<26} {'rho':>6} {'cov':>5} {'shard_GiB':>9} "
          f"{'search_ms(p50)':>14} {'search_ms(p90)':>14} {'svc/s':>6}", flush=True)
    for part in args.parts.split(","):
        meta = json.load(open(os.path.join(part, "partition_meta.json")))
        eng = VLiteEngine(os.environ["SR1_INDEX"], part, os.environ["SR1_E5"],
                          nprobe=args.nprobe, topk=args.topk, ngpu=1, enable_swap=False)
        qv = eng.encode(qs)
        # warm
        eng.search_vectors(qv[:args.batch], args.topk)
        lat = []
        for i in range(args.reps):
            sub = qv[(i * args.batch) % max(1, len(qv) - args.batch):][:args.batch]
            if sub.shape[0] < args.batch:
                sub = qv[:args.batch]
            t = time.time()
            eng.search_vectors(sub, args.topk)
            lat.append((time.time() - t) * 1000)
        lat = np.array(lat)
        p50, p90 = np.percentile(lat, 50), np.percentile(lat, 90)
        svc = 1000.0 / p50 * args.batch     # serialized searches/s
        print(f"{os.path.basename(part):<26} {meta['rho']:>6.3f} "
              f"{meta['coverage']:>5.2f} {meta['hot_gb']:>9.2f} "
              f"{p50:>14.0f} {p90:>14.0f} {svc:>6.2f}", flush=True)
        del eng
        faiss.omp_set_num_threads(32)


if __name__ == "__main__":
    main()
