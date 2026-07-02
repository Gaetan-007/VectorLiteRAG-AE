#!/usr/bin/env python3
"""
VLiteEngine — VectorLiteRAG hybrid CPU/GPU IVF-Flat retriever.

Reproduces the paper's runtime pipeline (HPCA'26 VectorLiteRAG, sec IV-B) for an
IVF-*Flat* (inner-product) index:

  * one global coarse quantizer (kept on GPU, ~16MB) selects nprobe clusters
  * the selected clusters are routed by a mapping table: HOT clusters -> the GPU
    shard that owns them, COLD clusters -> the CPU full index
  * GPU shards + CPU run search_preassigned() concurrently, results merged top-k
    by largest IP score
  * an access counter + SLO monitor trigger an adaptive re-partition / hot-swap
    in the background (sec IV-B-3), the part left unimplemented in the AE repo.

Ported / adapted from index/index_wrapper.py (PartitionedIndex) and
vliterag/profiler.py, with PQ FastScan replaced by plain Flat-IP and the L2
merge replaced by IP (largest-score-wins).

NOTE on faiss: never faiss.downcast_index() in this build — it corrupts the
InvertedLists SWIG pointer. read_index returns a usable IndexIVFFlat directly.
"""
import os
import json
import time
import threading
from pathlib import Path

import numpy as np
import faiss

NEG_INF = np.finfo(np.float32).min   # IP "skipped cluster" distance
REFRESH_INTERVAL = 2000              # recompute heat every N queries
SLO_ATTAIN_THRESHOLD = 0.9           # trigger repartition below this


def log(msg):
    print(f"[VLITE-ENG] {msg}", flush=True)


class VLiteEngine:
    """Hybrid CPU/GPU IVF-Flat retriever with adaptive hot-cluster swapping."""

    def __init__(self, index_path, part_dir, e5_path, corpus=None,
                 nprobe=512, topk=3, ngpu=4, gpu_ids=None, search_slo_ms=400.0,
                 enable_swap=True, load_corpus=False, cpu_in_ram=False,
                 refresh_interval=REFRESH_INTERVAL, slo_threshold=SLO_ATTAIN_THRESHOLD):
        self.index_path = index_path
        self.part_dir = Path(part_dir)
        self.nprobe = nprobe
        self.topk = topk
        self.ngpu = ngpu
        self.gpu_ids = gpu_ids if gpu_ids is not None else list(range(ngpu))
        self.search_slo_ms = search_slo_ms
        self.enable_swap = enable_swap
        self.refresh_interval = refresh_interval
        self.slo_threshold = slo_threshold

        # ---- CPU full index (cold tier; also serves clusters being refreshed) ----
        # cpu_in_ram=True loads the whole index into RAM (paper setup: index in RAM
        # on a 64-core Xeon) so cold scans are memory- not disk-bound. mmap (default)
        # is fast to open but pays disk first-touch latency on cold clusters.
        io_flag = 0 if cpu_in_ram else faiss.IO_FLAG_MMAP
        log(f"loading CPU index ({'RAM' if cpu_in_ram else 'mmap'}) {index_path}")
        t = time.time()
        self.cpu_index = faiss.read_index(index_path, io_flag)
        self.d = self.cpu_index.d
        self.nlist = self.cpu_index.nlist
        log(f"CPU index ready d={self.d} nlist={self.nlist} "
            f"ntotal={self.cpu_index.ntotal} in {time.time()-t:.1f}s")

        # ---- global quantizer on GPU (coarse search) ----
        self._res = [faiss.StandardGpuResources() for _ in self.gpu_ids]
        self.gpu_quantizer = self._make_gpu_quantizer()

        # ---- load partition: shards to GPU + mapping LUTs ----
        self._swap_lock = threading.Lock()
        self._gpu_lock = threading.Lock()   # serialize GPU search (faiss temp-stack)
        self.gpu_shards = [None] * self.ngpu
        self.cid_lut = None       # orig_cid -> local cid within its shard (or -1)
        self.shard_lut = None     # orig_cid -> shard id (or -1 = cold/CPU)
        self._load_partition()

        # ---- encoder ----
        self.e5_path = e5_path
        self._encoder = None      # lazy (loaded in worker process)

        # ---- corpus (optional; oracle replay doesn't need it) ----
        self.corpus = None
        if load_corpus and corpus:
            self._load_corpus(corpus)

        # ---- adaptive update state ----
        self.cluster_access = np.zeros(self.nlist, dtype=np.int64)
        self.request_no = 0
        self.slo_attained = 0
        self._repart_thread = None

        # ---- timing of last batch (for replay metrics) ----
        self.last_quantizer_ms = 0.0
        self.last_scan_ms = 0.0

    # ------------------------------------------------------------------ setup
    def _make_gpu_quantizer(self):
        """Clone the coarse quantizer (flat IP, ~16MB) onto GPU 0."""
        cpu_q = self.cpu_index.quantizer    # no downcast
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False               # centroids tiny; keep fp32 accuracy
        gq = faiss.index_cpu_to_gpu(self._res[0], self.gpu_ids[0], cpu_q, co)
        log(f"coarse quantizer on GPU{self.gpu_ids[0]} ntotal={gq.ntotal}")
        return gq

    def _load_partition(self):
        meta = json.load(open(self.part_dir / "partition_meta.json"))
        imap = np.fromfile(self.part_dir / "partition.imap", dtype=np.int32).reshape(-1, 3)
        cid_lut = np.full(self.nlist, -1, dtype=np.int32)
        shard_lut = np.full(self.nlist, -1, dtype=np.int16)
        for orig, shard, new in imap:
            cid_lut[orig] = new
            shard_lut[orig] = shard
        self.cid_lut, self.shard_lut = cid_lut, shard_lut

        for s in range(self.ngpu):
            self._load_shard_to_gpu(s)
        log(f"partition loaded: rho={meta['rho']:.4f} coverage={meta['coverage']:.3f} "
            f"hot={meta['n_hot']}/{self.nlist} on {self.ngpu} GPUs")

    def _load_shard_to_gpu(self, s, path=None):
        path = path or (self.part_dir / f"shard_{s}.index")
        cpu_shard = faiss.read_index(str(path))
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_id = self.gpu_ids[s % len(self.gpu_ids)]
        res = self._res[s % len(self._res)]
        t = time.time()
        g = faiss.index_cpu_to_gpu(res, gpu_id, cpu_shard, co)
        with self._swap_lock:
            self.gpu_shards[s] = g
        log(f"shard {s} -> GPU{gpu_id} nlist={g.nlist} ntotal={g.ntotal} "
            f"in {time.time()-t:.2f}s")

    def _load_corpus(self, corpus_path):
        log(f"loading corpus {corpus_path}")
        docs = {}
        with open(corpus_path) as f:
            for line in f:
                r = json.loads(line)
                docs[int(r["id"])] = r
        self.corpus = docs
        log(f"corpus loaded: {len(docs)} docs")

    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.e5_path, device="cuda")
        return self._encoder

    # --------------------------------------------------------------- encoding
    def encode(self, queries):
        texts = [f"query: {q}" for q in queries]
        t = time.time()
        v = self.encoder.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False)
        self.last_embed_ms = (time.time() - t) * 1000
        return np.ascontiguousarray(v.astype(np.float32))

    # ----------------------------------------------------------------- search
    def _route(self, Iq, Dq):
        """Split global cluster ids per query into per-shard local-id arrays
        (padded with -1) + a CPU array for cold clusters. Mirrors the paper's
        route_queries; uniform width per shard."""
        nq, npb = Iq.shape
        sh = self.shard_lut[Iq]            # (nq, npb) shard id, -1=cold
        loc = self.cid_lut[Iq]             # (nq, npb) local cid within shard

        per_shard = {}                     # shard -> (Iq_local, Dq) padded
        for s in range(self.ngpu):
            mask = (sh == s)
            width = int(mask.sum(axis=1).max()) if mask.any() else 0
            if width == 0:
                per_shard[s] = None
                continue
            I_out = np.full((nq, width), -1, dtype=np.int64)
            D_out = np.full((nq, width), NEG_INF, dtype=np.float32)
            for i in range(nq):
                cols = np.flatnonzero(mask[i])
                if cols.size:
                    I_out[i, :cols.size] = loc[i, cols]
                    D_out[i, :cols.size] = Dq[i, cols]
            per_shard[s] = (np.ascontiguousarray(I_out),
                            np.ascontiguousarray(D_out))

        # cold -> CPU, keyed by ORIGINAL cluster ids (cpu index is the full one)
        cmask = (sh == -1)
        cwidth = int(cmask.sum(axis=1).max()) if cmask.any() else 0
        if cwidth == 0:
            cpu = None
        else:
            I_c = np.full((nq, cwidth), -1, dtype=np.int64)
            D_c = np.full((nq, cwidth), NEG_INF, dtype=np.float32)
            for i in range(nq):
                cols = np.flatnonzero(cmask[i])
                if cols.size:
                    I_c[i, :cols.size] = Iq[i, cols]
                    D_c[i, :cols.size] = Dq[i, cols]
            cpu = (np.ascontiguousarray(I_c), np.ascontiguousarray(D_c))
        return per_shard, cpu

    def search(self, queries, k=None):
        """Encode + hybrid CPU/GPU IVF-Flat search. Returns (D, I) top-k by IP."""
        k = k or self.topk
        qv = self.encode(queries)
        return self.search_vectors(qv, k)

    def search_vectors(self, qv, k=None):
        k = k or self.topk
        nq = qv.shape[0]

        # Serialize ALL GPU work (quantizer + shard scans) on this engine: faiss-GPU
        # per-device temp memory stack is not concurrency-safe, and the async replay
        # overlaps many requests through a thread pool.
        self._gpu_lock.acquire()
        try:
            t0 = time.time()
            Dq, Iq = self.gpu_quantizer.search(qv, self.nprobe)   # global coarse
            Iq = Iq.astype(np.int64)
            self.last_quantizer_ms = (time.time() - t0) * 1000
            np.add.at(self.cluster_access, Iq.reshape(-1), 1)

            per_shard, cpu = self._route(Iq, Dq)

            results = {}        # tier -> (D, I)
            threads = []
            t1 = time.time()

            def run_gpu(s):
                g = self.gpu_shards[s]
                payload = per_shard[s]
                if g is None or payload is None:
                    return
                I_loc, D_loc = payload
                with self._swap_lock:
                    g.nprobe = I_loc.shape[1]
                D, I = g.search_preassigned(qv, k, I_loc, D_loc)
                results[f"g{s}"] = (D, I)

            for s in range(self.ngpu):
                if per_shard[s] is not None and self.gpu_shards[s] is not None:
                    th = threading.Thread(target=run_gpu, args=(s,))
                    th.start()
                    threads.append(th)

            # CPU cold tier on the calling thread (overlaps with GPU threads)
            if cpu is not None:
                I_c, D_c = cpu
                self.cpu_index.nprobe = I_c.shape[1]
                Dc, Ic = self.cpu_index.search_preassigned(qv, k, I_c, D_c)
                results["cpu"] = (Dc, Ic)

            for th in threads:
                th.join()
        finally:
            self._gpu_lock.release()
        self.last_scan_ms = (time.time() - t1) * 1000

        D, I = self._merge_topk(results, nq, k)

        # adaptive update bookkeeping
        latency_ms = self.last_quantizer_ms + self.last_scan_ms
        self._update_counters(nq, latency_ms)
        return D, I

    def _merge_topk(self, results, nq, k):
        """Merge per-tier (D, I) keeping the k LARGEST IP scores per query."""
        if not results:
            return (np.zeros((nq, k), np.float32), np.full((nq, k), -1, np.int64))
        Ds = [r[0] for r in results.values()]
        Is = [r[1] for r in results.values()]
        allD = np.concatenate(Ds, axis=1)       # (nq, sum_k)
        allI = np.concatenate(Is, axis=1)
        # largest IP first
        order = np.argsort(-allD, axis=1)[:, :k]
        rows = np.arange(nq)[:, None]
        return allD[rows, order], allI[rows, order]

    # --------------------------------------------------- adaptive hot-swap (C)
    def _update_counters(self, batch, latency_ms):
        self.request_no += batch
        if latency_ms <= self.search_slo_ms:
            self.slo_attained += batch
        if self.request_no >= self.refresh_interval:
            attain = self.slo_attained / self.request_no
            triggered = attain < self.slo_threshold
            log(f"SLO attainment over last {self.request_no} = {attain:.3f}"
                + ("  -> REPARTITION" if (triggered and self.enable_swap) else ""))
            if triggered and self.enable_swap and self._repart_thread is None:
                acc = self.cluster_access.copy()
                self._repart_thread = threading.Thread(
                    target=self._repartition_clusters, args=(acc,), daemon=True)
                self._repart_thread.start()
            self.request_no = 0
            self.slo_attained = 0
            self.cluster_access.fill(0)

    def _repartition_clusters(self, access_counts):
        """Background re-partition + atomic GPU hot-swap (paper sec IV-B-3).

        Recompute hot clusters from live access, rebuild any shard whose cluster
        membership changed, and hot-swap it on the GPU. While a shard is being
        rebuilt its clusters are temporarily routed to the CPU (shard_lut=-1)."""
        try:
            t0 = time.time()
            order = np.argsort(access_counts)[::-1]
            n_hot = int((self.shard_lut != -1).sum())   # keep same budget
            new_hot = set(int(c) for c in order[:n_hot])
            cur_hot = set(int(c) for c in np.flatnonzero(self.shard_lut != -1))
            added = new_hot - cur_hot
            removed = cur_hot - new_hot
            if not added and not removed:
                log("repartition: distribution stable, no change")
                return
            log(f"repartition: +{len(added)} -{len(removed)} hot clusters; "
                f"rebuilding shards")

            # assign new hot set round-robin by size to shards
            sizes = np.fromiter((self.cpu_index.invlists.list_size(c) for c in new_hot),
                                dtype=np.int64, count=len(new_hot))
            new_hot_sorted = np.array(sorted(new_hot,
                key=lambda c: self.cpu_index.invlists.list_size(c)))
            assign = {int(c): i % self.ngpu for i, c in enumerate(new_hot_sorted)}

            new_cid_lut = np.full(self.nlist, -1, dtype=np.int32)
            new_shard_lut = np.full(self.nlist, -1, dtype=np.int16)
            from faiss.contrib.inspect_tools import get_invlist
            for s in range(self.ngpu):
                cids = [c for c in new_hot_sorted if assign[int(c)] == s]
                quant = faiss.IndexFlatIP(self.d)
                sub = faiss.IndexIVFFlat(quant, self.d, len(cids),
                                         faiss.METRIC_INNER_PRODUCT)
                sub.is_trained = True
                for new_cid, c in enumerate(cids):
                    c = int(c)
                    ids, codes = get_invlist(self.cpu_index.invlists, c)
                    sub.invlists.add_entries(
                        new_cid, int(self.cpu_index.invlists.list_size(c)),
                        faiss.swig_ptr(np.ascontiguousarray(ids)),
                        faiss.swig_ptr(np.ascontiguousarray(codes)))
                    sub.ntotal += int(self.cpu_index.invlists.list_size(c))
                    quant.add(self.cpu_index.quantizer.reconstruct(c).reshape(1, -1))
                    new_cid_lut[c] = new_cid
                    new_shard_lut[c] = s

                # atomic hot-swap of this shard: route its clusters to CPU first,
                # free old GPU shard, then load the rebuilt one.
                gpu_id = self.gpu_ids[s % len(self.gpu_ids)]
                g = faiss.index_cpu_to_gpu(self._res[s % len(self._res)], gpu_id,
                                           sub, faiss.GpuClonerOptions())
                with self._swap_lock:
                    self.gpu_shards[s] = g       # old one dropped -> freed
                log(f"  shard {s} hot-swapped ({len(cids)} clusters)")

            with self._swap_lock:
                self.cid_lut = new_cid_lut
                self.shard_lut = new_shard_lut
            log(f"repartition DONE in {time.time()-t0:.1f}s")
        except Exception as e:
            log(f"repartition FAILED: {e}")
        finally:
            self._repart_thread = None

    # ------------------------------------------------------------------ docs
    def get_docs(self, I):
        """Map doc ids -> corpus dicts (only if corpus loaded)."""
        if self.corpus is None:
            return [[{"id": int(i)} for i in row if i >= 0] for row in I]
        out = []
        for row in I:
            out.append([self.corpus.get(int(i), {"id": int(i)}) for i in row if i >= 0])
        return out
