# VectorLiteRAG — Search-R1 Replay Harness (Usage)

This document covers the **agentic Search-R1 replay harness** added under `tools/sr1/`
(and the runtime engine `vliterag/vlite_engine.py`). It is a self-contained,
single-node reproduction of the paper's three runtime optimizations —
hybrid CPU/GPU IVF-Flat search, latency-bounded (SLO-aware) hot-cluster
partitioning, an online hot-swap updater, and a dynamic dispatcher — driven by a
real multi-turn Search-R1 trace served through vLLM (Qwen3-4B).

It is **complementary to** the paper artifact documented in the top-level
[`README.md`](../README.md) (conda + `main.py` + `scripts/runall_*.sh`). The harness
here does **not** use conda and does **not** require building this repo's own FAISS
— it runs against a prebuilt GPU FAISS via `PYTHONPATH`, using only the stock
`search_preassigned` / `index_cpu_to_gpu` API.

> **Scope note.** The offline partitioning + online serving pipeline is
> distance-metric / compression agnostic. This harness targets the Search-R1
> `IVF4096,Flat` (inner-product) index; because Flat vectors are ~16× larger than
> IVFPQ, the GPU-resident hot set is memory-bound (see the memory trade-off notes
> at the end).

---

## 1. Prerequisites

| Asset | Default location (override via env var) |
|-------|------------------------------------------|
| Index | `SR1_INDEX` = `/mnt/public/Agent_index/wiki-25-corpus/e5_IVF4096,Flat.index` (IVF-Flat, d=1024, nlist=4096, ntotal≈20.6M, **inner-product**) |
| Corpus | `SR1_CORPUS` = `.../wiki-25-chunk-256.jsonl` |
| Retriever encoder | `SR1_E5` = `/mnt/public/multilingual-e5-large-instruct` |
| LLM | `SR1_LLM` = `/mnt/public/Qwen3-4B` |
| Trace | `SR1_TRACE` = `/mnt/public/wangzehao/AGInf-Mem/data/search-r1/records-hotpotqa.jsonl` |
| Python venv | repo-local `.venv` (py3.10), exposed as `$VLITE_PY` |

The environment scripts set `LD_LIBRARY_PATH` (nvidia wheel CUDA libs + MKL),
`PYTHONPATH` (FAISS + repo root), and the asset env vars above.

**Pick ONE FAISS backend and source it before every command:**

```bash
# Option A — prebuilt GPU FAISS via PYTHONPATH (default; nothing to build):
source tools/sr1/env.sh

# Option B — this repo's own custom-built FAISS (requires ./scripts/build.sh first,
# plus the .cudaenv toolchain — see README "Build native dependencies"):
source tools/sr1/env_ownfaiss.sh
```

Both scripts export `$VLITE_PY` (the venv interpreter). Run every tool as
`$VLITE_PY tools/sr1/<script>.py ...` so it uses the correct interpreter + paths.

---

## 2. Two-phase workflow

```
                (offline, once per GPU count / SLO)          (online, per experiment)
 index ──▶  vlite_partition.py  ──▶  database/sr1/<name>/  ──▶  replay.py  ──▶  runs/<name>/summary.json
                 │  builds N GPU shards (round-robin,             │  vLLM Qwen3-4B + hybrid retriever
                 │  balanced by size) + partition.imap            │  under a Poisson request arrival sweep
                 └─ chooses hot-cluster set ρ from either         └─ CPU-only baseline vs. vlite hybrid
                    a coverage target or an SLO (Algorithm 1)
```

- **Phase 1 — `vlite_partition.py` (offline):** profiles cluster "heat" from trace
  queries, chooses the hot-cluster fraction ρ, and writes `N` GPU shards
  (`shard_0.index … shard_{N-1}.index`), a routing map `partition.imap`,
  `partition_meta.json`, and `heat.npz`. **`--ngpu N` is the multi-GPU knob here.**
- **Phase 2 — `replay.py` (online):** loads the shards onto GPUs, co-locates the
  vLLM LLM on the same cards (tensor-parallel `--tp`), and replays the trace at each
  requested arrival rate, writing per-request records + a `summary.json`.

`database/` and `runs/` are git-ignored — only the harness **code** is version
controlled; the partition and result artifacts are regenerated per machine.

---

## 3. Multi-GPU configuration (the important part)

Two independent counts must agree, and they share the same physical GPUs
(retrieval shard + LLM are **co-located**, not on separate cards):

| Knob | Where | Meaning |
|------|-------|---------|
| `--ngpu N` | `vlite_partition.py` **and** `replay.py` | number of FAISS GPU shards. The partition builder round-robins hot clusters across `N` shards balanced by size; the engine loads shard `s` onto GPU `s % len(gpu_ids)`. |
| `--tp N` | `replay.py` | vLLM tensor-parallel degree. Set it equal to `--ngpu` so the LLM spans the same cards as the shards. |
| `--gpu_index_budget_gb G` | `vlite_partition.py` | **per-GPU** GiB reserved for the index shard. Larger ⇒ more hot clusters fit ⇒ higher GPU coverage, but less room left for vLLM. |
| `--gpu_mem_util F` | `replay.py` | fraction of each GPU vLLM may use for weights + KV cache. Must leave room for the shard: roughly `F ≲ (24GB − shard_GB − overhead) / 24GB` per card. |

GPU placement details (verified in `vliterag/vlite_engine.py`):
- The **coarse quantizer** (the flat IP quantizer carried by the loaded index, an
  `IndexFlatIP` for this inner-product index) is cloned onto the first GPU
  (`gpu_ids[0]`); it is tiny — for this index (nlist=4096, d=1024, fp32) that is
  nlist×d×4 ≈ 16 MB.
- Each hot shard `s` is loaded onto `gpu_ids[s % len(gpu_ids)]`. With `ngpu`
  shards and the default `gpu_ids = range(ngpu)`, that is one shard per GPU.
- All GPU work (quantizer + per-shard `search_preassigned`) is serialized behind a
  single lock per engine — FAISS-GPU's per-device temp-memory stack is not
  concurrency-safe under the async replay's parallel requests. Throughput comes
  from batching (the dispatcher), not from unsynchronized concurrency.
- Cold clusters (not in any shard) are scanned on the CPU full index concurrently
  and merged by inner-product top-k.

### 3a. Building a multi-GPU partition

```bash
source tools/sr1/env.sh
cat "$SR1_INDEX" > /dev/null          # warm the page cache (skips the 8–34 s cold first scan)

# 4-GPU, high-coverage partition (paper-style; needs ~6 GB/GPU for the shard):
$VLITE_PY tools/sr1/vlite_partition.py \
  --ngpu 4 \
  --target_coverage 0.85 \
  --gpu_index_budget_gb 6 \
  --out database/sr1/4gpus_hi
# -> ρ≈0.29 (1195/4096 hot clusters), coverage≈0.85, ~5.8 GiB/GPU shard.
```

**SLO-aware partitioning (paper Algorithm 1)** — derive ρ from a search-stage
latency target instead of a manual coverage number:

```bash
$VLITE_PY tools/sr1/vlite_partition.py \
  --ngpu 4 --search_slo_ms 400 --gpu_index_budget_gb 6 \
  --out database/sr1/4gpus_slo400
```

**SLO sensitivity sweep (paper Table II)** — profile once, print the ρ-vs-SLO table
without building shards:

```bash
$VLITE_PY tools/sr1/vlite_partition.py --ngpu 4 --gpu_index_budget_gb 16 \
  --slo_sweep 150,300,400,600,1000
```

### 3b. Running the multi-GPU replay

```bash
source tools/sr1/env.sh
cat "$SR1_INDEX" > /dev/null          # warm page cache first

# CPU-only baseline (retrieval fully on CPU; vLLM still tp=4):
$VLITE_PY tools/sr1/replay.py --retriever cpu \
  --rps_list 2,4 --limit 200 --tp 4 --gpu_mem_util 0.6 \
  --out_dir runs/sr1_cpu_full

# vlite hybrid (GPU shards + CPU cold tier). gpu_mem_util is LOWER here because
# the shards already occupy GPU memory alongside the LLM:
$VLITE_PY tools/sr1/replay.py --retriever vlite \
  --part_dir database/sr1/4gpus_hi \
  --ngpu 4 --tp 4 --nprobe 256 \
  --rps_list 2,4 --limit 200 --gpu_mem_util 0.4 \
  --out_dir runs/sr1_vlite_full
```

Add optimizations independently:
- `--dispatcher` — dynamic cross-request micro-batching (paper §IV-B-2). Tunable
  with `--disp_max_batch 16 --disp_window_ms 5`.
- `--enable_swap` — online adaptive hot-cluster hot-swap under drift (paper
  §IV-B-3); pairs with a deliberately stale partition (see §5).
- `--cpu_in_ram` — load the CPU cold-tier index into RAM instead of mmap (removes
  disk first-touch latency; needs enough system memory for the full index).

### 3c. Verified 4-GPU result (this harness, `nprobe=256`, 200 requests)

`4gpus_hi` partition, Qwen3-4B tp=4 co-located on the retrieval GPUs:

| rps | retriever | throughput | TTFT p90 (ms) | retrieval mean (ms) | SLO |
|----:|:----------|-----------:|--------------:|--------------------:|----:|
| 2 | CPU-only | 1.54 | 141 | 980 | 1.00 |
| 2 | vlite | 1.64 | 67 | 553 | 0.99 |
| 4 | CPU-only | 1.91 | 134 | 1068 | 1.00 |
| 4 | vlite | 2.48 | 72 | **95** | 1.00 |

At rps=4 the CPU retriever saturates (retrieval ≈1.07 s/query) while the hybrid
stays ≈95 ms — the paper's core claim: under load, CPU retrieval degrades and the
GPU-hot tier keeps the SLO-compliant request rate wide.

---

## 4. Single-GPU configuration

Everything above collapses to `--ngpu 1 --tp 1` on a single card. Because one 24 GB
card must hold the shard **and** the LLM (weights + KV), the shard budget is small
and the LLM's `gpu_mem_util` must be tuned down accordingly.

```bash
source tools/sr1/env.sh
cat "$SR1_INDEX" > /dev/null

# Build a single-GPU partition (3 GB shard budget):
$VLITE_PY tools/sr1/vlite_partition.py \
  --ngpu 1 --target_coverage 0.35 --gpu_index_budget_gb 3 \
  --out database/sr1/1gpu_3g

# Replay on one card (pin the device explicitly):
CUDA_VISIBLE_DEVICES=0 $VLITE_PY tools/sr1/replay.py --retriever vlite \
  --part_dir database/sr1/1gpu_3g \
  --ngpu 1 --tp 1 --nprobe 512 --topk 3 \
  --rps_list 0.5,1,2,3 --limit 200 --gpu_mem_util 0.6 \
  --dispatcher --out_dir runs/1gpu_vlite
```

**Isolated per-point sweep (recommended for clean curves).**
`collect_points.py` launches one fresh vLLM process **per rps point** (no
cross-point interference), auto-selects the freest GPU via `nvidia-smi` and waits
until it has enough free memory, and re-runs a point if it detects an anomaly
(short run / OOM / engine death / out-of-band retrieval latency):

```bash
# NOTE: argparse needs the "=" form when the list starts with a negative (burst) point:
$VLITE_PY tools/sr1/collect_points.py \
  --rps=-1,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5 \
  --part_dir database/sr1/1gpu_3g --limit 200 --gpu_mem_util 0.6 \
  --out_dir runs/vlite_full

# Aggregate the per-point summaries into a Markdown + CSV table.
# NOTE: aggregate.py takes NO arguments — it reads the hardcoded directory
# runs/vlite_full (globbing rps*/summary.json) and writes RESULTS.md + RESULTS.csv
# there. Point --out_dir at runs/vlite_full above to feed it.
$VLITE_PY tools/sr1/aggregate.py
```

Single-4090 headline: the SLO-compliant arrival rate extends from ~1 req/s
(CPU-only) to ~3 req/s (vlite), reproducing the paper claim on one GPU. On a single
card the best trade-off is a *moderate* shard (≈cov-0.35, `gpu_mem_util≈0.53`):
retrieval is the bottleneck, so extra hot-cluster coverage beats a bigger KV cache
— but too large a shard starves the LLM KV and degrades throughput.

---

## 5. Adaptive hot-swap under drift (paper §IV-B-3)

Build a deliberately misaligned ("stale") partition and let the online updater
realign the hot set at runtime:

```bash
# stale partition: hot set drawn from the WRONG cluster tier
$VLITE_PY tools/sr1/vlite_partition.py --ngpu 4 --gpu_index_budget_gb 6 \
  --stale_offset 1500 --out database/sr1/4gpus_stale

# drift demo: --enable_swap lets the engine detect the SLO miss and repartition in a
# background thread — it rebuilds each shard from the live access counts and swaps
# the routing LUTs (cid_lut/shard_lut) atomically once all shards are rebuilt, so
# service is uninterrupted (queries keep hitting the existing shards until the swap).
$VLITE_PY tools/sr1/drift_demo.py --part_dir database/sr1/4gpus_stale --ngpu 4
```

---

## 6. Key operational gotchas

- **Warm the page cache** (`cat "$SR1_INDEX" > /dev/null`) before any run; the first
  cold mmap scan of the 20 GB index costs 8–34 s and pollutes latency numbers.
- **Never `faiss.downcast_index()`** in these FAISS builds — it corrupts the
  inverted-list SWIG pointer (garbage `nlist`, segfault). `read_index` already
  returns a usable `IndexIVFFlat`.
- **`search_preassigned` requires** the assigned-cluster width to equal
  `index.nprobe`; `-1` is valid padding for cold/absent clusters.
- **Transformers pin:** vLLM 0.9.1 needs `transformers==4.53.2` (newer versions
  clash on an `aimv2` config). One `asyncio.run` wraps the whole rps sweep — a fresh
  loop per level kills the `AsyncLLMEngine` with `EngineDeadError`.
- **Crashed runs orphan vLLM workers** holding GPU memory — `pkill -9 VllmWorker`
  before restarting.
- **`--gpu_mem_util` interplay:** the FAISS footprint on a card (shard + quantizer +
  temp memory + the e5 encoder) is larger than the shard file alone. If vLLM OOMs,
  lower `--gpu_mem_util` or shrink the shard (`--gpu_index_budget_gb`).

## 7. Memory / index-type trade-off

The hybrid mechanism is compression-agnostic, but per-vector memory drives how much
of the index can be GPU-resident. IVF-**Flat** stores `d×4` bytes/vector; IVF**PQ**
(4-bit) stores ~`M×0.5` bytes/vector — roughly a 16× difference. On memory-limited
cards a pure-Flat hot set is small, so at high `nprobe` many probes still fall to the
CPU cold tier. If you need higher recall without the Flat blow-up, an SQ8 or 8-bit PQ
index is a middle ground (2–4× the PQ memory, hot clusters still fit GPU).
