#!/usr/bin/env python3
"""
VectorLiteRAG offline partitioner for the Search-R1 wiki-25 IVF-Flat index.

Reproduces the paper's "Access-Skew-Aware Data Layout" + "Latency-Bounded
Partitioning" (HPCA'26 VectorLiteRAG, sec IV-A) adapted to an IVF *Flat* (IP)
index instead of IVF-PQ:

  1. Cluster heat: encode a sample of Search-R1 queries with e5, run the coarse
     quantizer, count per-cluster access frequency -> ordered centroids + CDF.
  2. Latency-bounded partition point rho: pick the smallest set of HOT clusters
     whose CDF coverage >= target, subject to a GPU memory budget (per-GPU bytes
     reserved for the index, the rest left for vLLM KV cache).
  3. Split hot clusters round-robin (by size) into N GPU shards, each a small
     IndexIVFFlat (IP) carrying only its clusters. Cold clusters stay on the CPU
     index. Write cid_lut / shard_lut mapping for the runtime router.

Outputs into  database/sr1/<ngpu>gpus/ :
    shard_<i>.index        (i in 0..N-1)   GPU sub-indexes (IVFFlat IP)
    partition.imap         int32 [(orig_cid, shard, new_cid)] for hot clusters
    partition_meta.json    rho, coverage, ngpu, nlist, hot/cold counts, budget

Run:  source tools/sr1/env.sh && $VLITE_PY tools/sr1/vlite_partition.py --help
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import faiss
from faiss.contrib.inspect_tools import get_invlist_sizes, get_invlist


def log(msg):
    print(f"[VLITE-PART] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. cluster heat from Search-R1 queries
# --------------------------------------------------------------------------
def load_trace_queries(trace_path, max_q):
    """Pull search queries from a Search-R1 trace. Uses the top-level question
    plus any <search> queries embedded in trace_turns' llm_output."""
    import re
    search_re = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    qs = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("question"):
                qs.append(rec["question"])
            for t in rec.get("trace_turns", []) or []:
                for m in search_re.findall(t.get("llm_output", "") or ""):
                    m = m.strip()
                    if m:
                        qs.append(m)
            if len(qs) >= max_q:
                break
    return qs[:max_q]


def encode_queries(e5_path, queries, batch=128):
    from sentence_transformers import SentenceTransformer
    log(f"loading e5 from {e5_path}")
    m = SentenceTransformer(e5_path, device="cuda")
    # e5 query convention
    texts = [f"query: {q}" for q in queries]
    vecs = m.encode(texts, batch_size=batch, normalize_embeddings=True,
                    show_progress_bar=False)
    return np.ascontiguousarray(vecs.astype(np.float32))


def cluster_heat(quantizer, qvecs, nprobe, nlist):
    """Return ordered_cids (hot->cold), cumulative coverage CDF, and the per-query
    hit-rate variance measured at the 0.5-CDF cache point (max_var for the Beta
    model, paper sec IV-A-2 / profiler.py HitRateEstimator.collect_centroid_data)."""
    _, Iq = quantizer.search(qvecs, nprobe)
    freq = np.bincount(Iq.reshape(-1), minlength=nlist).astype(np.int64)
    order = np.argsort(freq)[::-1]                       # hot -> cold
    sorted_freq = freq[order]
    cdf = np.cumsum(sorted_freq) / max(1, sorted_freq.sum())

    # variance of per-query hit rate when caching the hot prefix covering ~0.5
    half = int(np.argmax(cdf >= 0.5))
    cached = order[:max(half, 1)]
    mask = np.isin(Iq, cached)
    hit_rates = mask.sum(axis=1).astype(np.float32) / nprobe
    max_var = float(np.var(hit_rates))
    return order.astype(np.int64), cdf, max_var


# --------------------------------------------------------------------------
# 2. latency-bounded partition point (memory-budgeted)
# --------------------------------------------------------------------------
def choose_partition_point(order, cdf, list_sizes, d, ngpu,
                           gpu_index_budget_gb, target_coverage):
    """Pick the largest hot-cluster prefix that (a) reaches target_coverage and
    (b) fits within ngpu * gpu_index_budget_gb of GPU memory.

    Flat vector bytes = ntotal_hot * d * 4. This is the paper's memory term with
    PQ replaced by full float32 (sec: 'independent of compression method')."""
    budget_bytes = ngpu * gpu_index_budget_gb * (1024 ** 3)
    cum_vecs = np.cumsum(list_sizes[order].astype(np.int64))
    bytes_at = cum_vecs * d * 4

    # largest prefix under the memory budget
    fit = np.searchsorted(bytes_at, budget_bytes, side="right")
    # prefix reaching the desired coverage
    cov = int(np.searchsorted(cdf, target_coverage, side="left")) + 1
    n_hot = int(min(fit, cov, len(order)))
    n_hot = max(n_hot, 0)
    return n_hot


# --------------------------------------------------------------------------
# 2b. SLO-aware latency-bounded partition point (paper sec IV-A-3, Algorithm 1)
# --------------------------------------------------------------------------
class _BetaHitRate:
    """Tail hit-rate model (paper sec IV-A-2), a minimal self-contained copy of
    vliterag/profiler.py HitRateEstimator's Beta math (imported directly would pull
    in pyarrow via vliterag.runner). var(eta) ~= 4*max_var*eta*(1-eta) [eq. before
    eq.2]; expected min hit rate of a batch = first-order statistic [eq.2]."""
    def __init__(self, nprobe, max_var=0.25):
        self.nprobe = nprobe
        self.max_var = max_var

    def _alpha_beta(self, mean):
        var = 4 * self.max_var * mean * (1 - mean)
        common = mean * (1 - mean) / var - 1
        return mean * common, (1 - mean) * common

    def min_for_mean(self, batch_size, mean):
        """eq.2: expected minimum hit rate within a batch given mean hit rate."""
        if mean >= 1.0:
            return 1.0
        if mean <= 0.0:
            return 0.0
        from scipy.stats import beta
        from scipy.integrate import quad
        a, b = self._alpha_beta(mean)

        def integrand(x):
            f = beta.pdf(x, a, b)
            F = beta.cdf(x, a, b)
            return x * batch_size * ((1 - F) ** (batch_size - 1)) * f
        val, _ = quad(integrand, 0, 1)
        return val

    def mean_for_min(self, exp_min, batch_size, tol=1e-3):
        """Invert eq.2: required MEAN hit rate to achieve a target MIN hit rate."""
        if exp_min <= 0.0 or exp_min >= 1.0:
            return exp_min
        lo, hi, mid = 0.0, 1.0, exp_min
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.min_for_mean(batch_size, mid) < exp_min:
                lo = mid
            else:
                hi = mid
        return mid


def choose_partition_point_slo(order, cdf, max_var, list_sizes, d, ngpu,
                               gpu_index_budget_gb, search_slo_ms, nprobe,
                               per_cluster_ms, quant_ms, batch_size, eps=1.0):
    """Reproduce the paper's latency-bounded partitioning for IVF-Flat.

    Hybrid search latency model (paper eq.1, PQ-LUT replaced by Flat cold-scan):
        tau_s(b) = T_quant(b) + (1 - eta) * T_coldscan(b)
    where T_coldscan(b) = per_cluster_ms * b * nprobe  is the cost of scanning ALL
    nprobe clusters on the CPU, and eta is the min hit rate within the batch.

    Algorithm 1: target serving time tau_s = SLO/(1+eps) (eps=1 -> SLO/2). Solve
    for the minimum hit rate eta that meets tau_s, convert eta(min) -> mean hit
    rate via the Beta first-order-statistic model (eq.2), then mean hit rate ->
    coverage rho via the access CDF. Finally clamp rho to the GPU memory budget.
    Returns (n_hot, rho, coverage, eta_min, mean_hr, slo_feasible)."""
    nlist = len(order)
    tau_s = (search_slo_ms / 1000.0) / (1.0 + eps)          # serving-time target (s)
    b = max(int(batch_size), 1)
    t_quant = quant_ms.get(b, quant_ms.get(1, 1.0)) / 1000.0
    t_coldscan = per_cluster_ms / 1000.0 * b * nprobe       # scan ALL nprobe on CPU (s)

    # eq.1 solved for eta: (T_quant + (1-eta)*T_cold) <= tau_s
    # -> eta_min >= 1 - (tau_s - T_quant)/T_cold
    eta_min = float(np.clip(1.0 - (tau_s - t_quant) / max(t_coldscan, 1e-9), 0.0, 1.0))

    # invert Beta first-order statistic: min hit rate -> required MEAN hit rate
    hest = _BetaHitRate(nprobe, max_var=max(max_var, 1e-4))
    mean_hr = hest.mean_for_min(eta_min, b)

    # mean hit rate -> coverage rho via the access CDF (smallest prefix covering it)
    if mean_hr <= 0.0:
        n_cov = 0
    elif mean_hr >= 1.0:
        n_cov = nlist
    else:
        n_cov = int(np.searchsorted(cdf, mean_hr, side="left")) + 1

    # clamp to GPU memory budget
    budget_bytes = ngpu * gpu_index_budget_gb * (1024 ** 3)
    cum_vecs = np.cumsum(list_sizes[order].astype(np.int64))
    bytes_at = cum_vecs * d * 4
    fit = int(np.searchsorted(bytes_at, budget_bytes, side="right"))

    n_hot = int(min(max(n_cov, 0), fit, nlist))
    slo_feasible = n_cov <= fit          # could we meet SLO within the mem budget?
    rho = n_hot / nlist
    coverage = float(cdf[n_hot - 1]) if n_hot > 0 else 0.0
    return n_hot, rho, coverage, eta_min, mean_hr, slo_feasible


# --------------------------------------------------------------------------
# 3. split hot clusters into GPU shards (IVFFlat IP)
# --------------------------------------------------------------------------
def build_shards(ivf, hot_cids, list_sizes, ngpu, d, out_dir):
    """Round-robin hot clusters across ngpu shards (balanced by size), each a
    standalone IndexIVFFlat(IP). Returns mapping rows (orig, shard, new_cid)."""
    quantizer_full = ivf.quantizer       # no downcast: corrupts SWIG invlists ptr in this build
    invlists = ivf.invlists

    # sort hot clusters by size, deal round-robin so shards are balanced
    hot = np.asarray(hot_cids, dtype=np.int64)
    hot = hot[np.argsort(list_sizes[hot])]
    shard_of = {int(c): (i % ngpu) for i, c in enumerate(hot)}

    rows = []
    for s in range(ngpu):
        cids = [c for c in hot if shard_of[int(c)] == s]
        quant = faiss.IndexFlatIP(d)
        sub = faiss.IndexIVFFlat(quant, d, len(cids), faiss.METRIC_INNER_PRODUCT)
        sub.is_trained = True
        quant.is_trained = True
        for new_cid, cid in enumerate(cids):
            cid = int(cid)
            ids, codes = get_invlist(invlists, cid)
            sz = int(list_sizes[cid])
            sub.invlists.add_entries(new_cid, sz,
                                     faiss.swig_ptr(np.ascontiguousarray(ids)),
                                     faiss.swig_ptr(np.ascontiguousarray(codes)))
            sub.ntotal += sz
            cvec = quantizer_full.reconstruct(cid).reshape(1, -1)
            quant.add(cvec)
            rows.append((cid, s, new_cid))
        path = out_dir / f"shard_{s}.index"
        faiss.write_index(sub, str(path))
        log(f"shard {s}: {len(cids)} clusters, ntotal={sub.ntotal}, -> {path}")
    return rows


def measure_coldscan_latency(ivf, qvecs, nprobe, batch_sizes=(1, 8, 32), reps=3):
    """Profile the CPU cold-scan latency model (paper sec IV-A-1): coarse-quantizer
    time and per-(cluster,query) scan cost vs batch size. Returns
    (quant_ms_by_b, per_cluster_ms) used by the latency-bounded partitioner."""
    import faiss as _f
    q = ivf.quantizer
    quant_ms = {}
    per_cluster = []
    for b in batch_sizes:
        sub = np.ascontiguousarray(qvecs[:b])
        q.search(sub, nprobe)                                  # warm
        t = time.time()
        for _ in range(reps):
            Dq, Iq = q.search(sub, nprobe)
        quant_ms[b] = (time.time() - t) / reps * 1000
        Iq = np.ascontiguousarray(Iq.astype(np.int64)); Dq = np.ascontiguousarray(Dq)
        ivf.nprobe = nprobe
        ivf.search_preassigned(sub, 3, Iq, Dq)                 # warm
        t = time.time()
        for _ in range(reps):
            ivf.search_preassigned(sub, 3, Iq, Dq)
        scan_ms = (time.time() - t) / reps * 1000
        per_cluster.append(scan_ms / (b * nprobe))
        log(f"  latency model b={b}: quant={quant_ms[b]:.1f}ms scan={scan_ms:.0f}ms "
            f"per-cluster-per-q={per_cluster[-1]:.3f}ms")
    return quant_ms, float(np.median(per_cluster))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.environ.get("SR1_INDEX"))
    ap.add_argument("--trace", default=os.environ.get("SR1_TRACE"))
    ap.add_argument("--e5", default=os.environ.get("SR1_E5"))
    ap.add_argument("--out", default=None, help="output dir (default database/sr1/<ngpu>gpus)")
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--nprobe", type=int, default=512)
    ap.add_argument("--sample_q", type=int, default=4000,
                    help="number of sample queries for heat profiling")
    ap.add_argument("--gpu_index_budget_gb", type=float, default=3.0,
                    help="per-GPU GiB reserved for the index (rest left for vLLM)")
    ap.add_argument("--target_coverage", type=float, default=0.5,
                    help="(legacy/manual mode) desired CDF coverage of hot clusters")
    ap.add_argument("--search_slo_ms", type=float, default=0.0,
                    help="if >0, use paper's latency-bounded partitioning (Algorithm 1) "
                         "to pick rho from this search-stage SLO instead of target_coverage")
    ap.add_argument("--exp_batch_size", type=int, default=8,
                    help="expected retrieval batch size for the latency model")
    ap.add_argument("--slo_sweep", default="",
                    help="comma-separated SLOs (ms); profile once and print the rho "
                         "table (paper Table II sensitivity) without building shards")
    ap.add_argument("--stale_offset", type=int, default=0,
                    help="P2 drift demo: build a deliberately misaligned partition using "
                         "clusters ranked [offset:offset+n_hot] instead of the true top-n_hot")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out) if args.out else root / "database" / "sr1" / f"{args.ngpu}gpus"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"reading index (mmap) {args.index}")
    # IMPORTANT: do NOT faiss.downcast_index() here — in this faiss build it
    # corrupts the InvertedLists SWIG pointer (invlists.nlist returns garbage and
    # segfaults). read_index already returns a usable IndexIVFFlat.
    ivf = faiss.read_index(args.index, faiss.IO_FLAG_MMAP)
    d, nlist = ivf.d, ivf.nlist
    list_sizes = get_invlist_sizes(ivf.invlists).astype(np.int64)
    log(f"IndexIVFFlat d={d} nlist={nlist} ntotal={ivf.ntotal} metric=IP")

    log(f"loading <= {args.sample_q} trace queries from {args.trace}")
    queries = load_trace_queries(args.trace, args.sample_q)
    log(f"got {len(queries)} queries; encoding with e5")
    qvecs = encode_queries(args.e5, queries)

    quant = ivf.quantizer        # no downcast (see note above)
    order, cdf, max_var = cluster_heat(quant, qvecs, args.nprobe, nlist)
    log(f"heat: top-20% clusters cover {cdf[int(0.2*nlist)]:.3f} of accesses; "
        f"hit-rate var@0.5CDF={max_var:.4f}")

    # SLO sweep: profile latency once, print rho table (paper Table II), no shards
    if args.slo_sweep:
        quant_ms, per_cluster_ms = measure_coldscan_latency(ivf, qvecs, args.nprobe)
        log("=== SLO sensitivity (Algorithm 1) ===")
        log(f"{'SLO(ms)':>8} {'eta_min':>8} {'mean_hr':>8} {'rho':>7} {'coverage':>9} "
            f"{'hot_GiB/GPU':>11} {'feasible':>9}")
        for slo in [float(x) for x in args.slo_sweep.split(",")]:
            n_h, rho_, cov_, em, mh, feas = choose_partition_point_slo(
                order, cdf, max_var, list_sizes, d, args.ngpu, args.gpu_index_budget_gb,
                slo, args.nprobe, per_cluster_ms, quant_ms, args.exp_batch_size)
            gib = int(list_sizes[order[:n_h]].sum()) * d * 4 / (1024**3) / args.ngpu
            log(f"{slo:>8.0f} {em:>8.3f} {mh:>8.3f} {rho_:>7.4f} {cov_:>9.3f} "
                f"{gib:>11.2f} {str(feas):>9}")
        log(f"DONE (sweep only) in {time.time()-t0:.1f}s")
        return

    slo_meta = {}
    if args.search_slo_ms > 0:
        # paper's latency-bounded partitioning (Algorithm 1)
        log(f"profiling CPU cold-scan latency model (SLO={args.search_slo_ms}ms)")
        quant_ms, per_cluster_ms = measure_coldscan_latency(ivf, qvecs, args.nprobe)
        n_hot, rho, coverage, eta_min, mean_hr, feasible = choose_partition_point_slo(
            order, cdf, max_var, list_sizes, d, args.ngpu, args.gpu_index_budget_gb,
            args.search_slo_ms, args.nprobe, per_cluster_ms, quant_ms, args.exp_batch_size)
        log(f"SLO-bounded: eta_min={eta_min:.3f} -> mean_hr={mean_hr:.3f} -> "
            f"rho={rho:.4f} ({n_hot}/{nlist}) coverage={coverage:.3f} "
            f"feasible_within_mem={feasible}")
        slo_meta = dict(search_slo_ms=args.search_slo_ms, eta_min=eta_min,
                        mean_hr=mean_hr, max_var=max_var, per_cluster_ms=per_cluster_ms,
                        exp_batch_size=args.exp_batch_size, slo_feasible=bool(feasible),
                        mode="slo_bounded")
    else:
        n_hot = choose_partition_point(order, cdf, list_sizes, d, args.ngpu,
                                       args.gpu_index_budget_gb, args.target_coverage)
        rho = n_hot / nlist
        coverage = float(cdf[n_hot - 1]) if n_hot > 0 else 0.0
        slo_meta = dict(target_coverage=args.target_coverage, max_var=max_var,
                        mode="coverage_budget")

    hot_cids = order[:n_hot]
    if args.stale_offset > 0:
        # deliberately misaligned partition for the P2 drift demo: use the tier of
        # clusters ranked [offset : offset+n_hot] instead of the true top-n_hot, so
        # initial hit-rate is poor and the online updater must realign the hot set.
        lo = args.stale_offset
        hi = min(lo + n_hot, nlist)
        hot_cids = order[lo:hi]
        coverage = float(cdf[hi - 1] - cdf[lo - 1]) if lo > 0 else float(cdf[hi - 1])
        log(f"STALE partition: using clusters ranked [{lo}:{hi}] "
            f"(marginal coverage ~{coverage:.3f}) to simulate drifted distribution")
        slo_meta["stale_offset"] = int(lo)
    hot_vecs = int(list_sizes[hot_cids].sum())
    hot_gb = hot_vecs * d * 4 / (1024 ** 3)
    log(f"partition point rho={rho:.4f}  hot_clusters={n_hot}/{nlist}  "
        f"coverage={coverage:.3f}  hot_vecs={hot_vecs} ({hot_gb:.2f} GiB total, "
        f"{hot_gb/args.ngpu:.2f} GiB/GPU)")

    rows = build_shards(ivf, hot_cids, list_sizes, args.ngpu, d, out_dir)

    imap = np.array(rows, dtype=np.int32)
    imap.tofile(out_dir / "partition.imap")
    meta = dict(rho=rho, coverage=coverage, ngpu=args.ngpu, nlist=int(nlist),
                d=int(d), ntotal=int(ivf.ntotal), nprobe=args.nprobe,
                n_hot=int(n_hot), hot_vecs=hot_vecs, hot_gb=hot_gb,
                gpu_index_budget_gb=args.gpu_index_budget_gb, metric="IP",
                index_path=args.index, **slo_meta)
    with open(out_dir / "partition_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    # save heat for runtime hot-swap warm start
    np.savez(out_dir / "heat.npz", order=order, cdf=cdf, list_sizes=list_sizes)
    log(f"wrote imap ({imap.shape}), meta, heat.npz to {out_dir}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
