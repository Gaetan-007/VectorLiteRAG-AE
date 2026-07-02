#!/usr/bin/env python3
"""
Self-contained Search-R1 replay driver for VectorLiteRAG (no HedraRAG dependency).

Replays a Search-R1 trace (records-*.jsonl) as a multi-turn agentic RAG workload:
each request runs turn-by-turn through vLLM (Qwen3-4B), and on every <search> turn
calls a pluggable retriever (CPU-only full IVF-Flat  vs  VLiteEngine hybrid CPU/GPU).
Retrieved docs are replaced by the trace's oracle <information> blocks so the LLM
path is identical across retrievers — isolating the retrieval latency / SLO effect
that VectorLiteRAG optimizes.

Poisson arrivals at a target --rps (sweep with --rps_list). Records per-request TTFT,
end-to-end latency, per-turn retrieval ms, and SLO attainment into summary.json.

Run:  source tools/sr1/env.sh
      $VLITE_PY tools/sr1/replay.py --retriever vlite --part_dir database/sr1/4gpus_hi \
          --rps_list 1,2,4 --out_dir runs/sr1_vlite
"""
import os
import re
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path

import numpy as np

SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
INFO_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def log(m):
    print(f"[SR1-REPLAY] {m}", flush=True)


# --------------------------------------------------------------------------
# retrievers (both expose .encode(list[str]) -> vecs and .search(list[str]))
# --------------------------------------------------------------------------
class CpuRetriever:
    """Baseline: full IVF-Flat (IP) scanned on CPU only."""
    def __init__(self, index_path, e5_path, nprobe, topk, cpu_in_ram=False):
        import faiss
        io = 0 if cpu_in_ram else faiss.IO_FLAG_MMAP
        log(f"[cpu] loading index ({'RAM' if cpu_in_ram else 'mmap'})")
        self.index = faiss.read_index(index_path, io)
        self.index.nprobe = nprobe
        self.topk = topk
        self.e5_path = e5_path
        self._enc = None
        self.last_quantizer_ms = 0.0
        self.last_scan_ms = 0.0

    @property
    def encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(self.e5_path, device="cuda")
        return self._enc

    def encode(self, queries):
        v = self.encoder.encode([f"query: {q}" for q in queries],
                                normalize_embeddings=True, show_progress_bar=False)
        return np.ascontiguousarray(v.astype(np.float32))

    def search(self, queries, k=None):
        k = k or self.topk
        qv = self.encode(queries)
        t = time.time()
        D, I = self.index.search(qv, k)
        self.last_scan_ms = (time.time() - t) * 1000
        self.last_quantizer_ms = 0.0
        return D, I


def build_retriever(args):
    if args.retriever == "cpu":
        return CpuRetriever(args.index, args.e5, args.nprobe, args.topk,
                            cpu_in_ram=args.cpu_in_ram)
    from vliterag.vlite_engine import VLiteEngine
    return VLiteEngine(args.index, args.part_dir, args.e5, nprobe=args.nprobe,
                       topk=args.topk, ngpu=args.ngpu, enable_swap=args.enable_swap,
                       search_slo_ms=args.search_slo_ms, cpu_in_ram=args.cpu_in_ram)


# --------------------------------------------------------------------------
# dynamic dispatcher (paper sec IV-B-2): coalesce concurrent per-request
# retrievals into cross-request micro-batches, forward each query as soon as the
# batch completes. With batch=1 (no dispatcher) each request pays full search
# latency serialized behind the GPU lock; batching amortizes GPU/quantizer cost
# and cuts head-of-line blocking, which is exactly the effect the paper measures.
# --------------------------------------------------------------------------
class RetrievalDispatcher:
    def __init__(self, retriever, loop, max_batch=16, window_s=0.005):
        self.retriever = retriever
        self.loop = loop
        self.max_batch = max_batch
        self.window_s = window_s
        self._pending = []          # list of (query, future)
        self._lock = asyncio.Lock()
        self._flush_task = None
        self.last_batch_sizes = []

    async def search(self, query):
        fut = self.loop.create_future()
        async with self._lock:
            self._pending.append((query, fut))
            if len(self._pending) >= self.max_batch:
                self._flush_now()
            elif self._flush_task is None:
                self._flush_task = self.loop.create_task(self._flush_after_window())
        return await fut

    async def _flush_after_window(self):
        await asyncio.sleep(self.window_s)
        async with self._lock:
            self._flush_now()

    def _flush_now(self):
        if not self._pending:
            return
        batch = self._pending
        self._pending = []
        if self._flush_task is not None:
            self._flush_task.cancel()
            self._flush_task = None
        queries = [q for q, _ in batch]
        futs = [f for _, f in batch]
        self.last_batch_sizes.append(len(queries))

        def do_batch():
            D, I = self.retriever.search(queries, self.retriever.topk)
            comp = getattr(self.retriever, "last_scan_ms", 0.0) + \
                   getattr(self.retriever, "last_quantizer_ms", 0.0)
            return D, I, comp

        async def run():
            try:
                D, I, comp = await self.loop.run_in_executor(None, do_batch)
                # forward each query's slice immediately (per-query dispatch)
                for j, f in enumerate(futs):
                    if not f.done():
                        f.set_result(comp)
            except Exception as e:
                for f in futs:
                    if not f.done():
                        f.set_exception(e)
        self.loop.create_task(run())



# --------------------------------------------------------------------------
# trace loading
# --------------------------------------------------------------------------
def load_trace(path, limit=0):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
            if limit and len(recs) >= limit:
                break
    recs.sort(key=lambda r: r.get("arrival_offset_s", 0.0))
    return recs


def turn_plan(rec, no_thinking=True):
    """Per turn: (completion_tokens, [search queries]).

    A "turn" ends in a retrieval iff a <search> tag is present in its llm_output;
    the recorded transcript also marks each performed retrieval with an oracle
    <information> block. Trace formats vary (<search>...</search> in llm_output,
    [search]...[/search] in reasoning_trace, or only <information> markers), so we
    anchor the number of retrievals on the more reliable num_searches / info-block
    count and synthesize the query text from the question (retrieval result is
    discarded — oracle docs are injected regardless).

    no_thinking=True (HedraRAG default replay mode): the <think>...</think> block is
    stripped from the expected per-turn output, and the generated token budget is
    derived from the STRIPPED text (HedraRAG: max_tokens = len(stripped)//3 + 64),
    not the full step_completion_tokens (which includes the long reasoning). This
    matches the /no_think replay: the model emits only <search>/<answer>, ~11x fewer
    tokens per turn than thinking mode."""
    turns = rec.get("trace_turns", []) or []
    infos = INFO_RE.findall(rec.get("reasoning_trace", "") or "")
    # explicit <search> queries if present in per-turn output
    explicit = []
    for t in turns:
        explicit.append([s.strip() for s in SEARCH_RE.findall(t.get("llm_output", "") or "")
                         if s.strip()])
    n_search = rec.get("num_searches", 0) or sum(len(e) for e in explicit) or len(infos)

    plan = []
    search_budget = n_search
    for i, t in enumerate(turns):
        raw = t.get("llm_output", "") or ""
        if no_thinking:
            stripped = THINK_RE.sub("", raw).lstrip()
            toks = len(stripped) // 3 + 64          # HedraRAG no-thinking token budget
        else:
            toks = int(t.get("step_completion_tokens", 0)) or 64
        # fire a retrieval at the end of each non-final turn (trace pattern:
        # n_turns == num_searches + 1), or within a lone turn that searched.
        is_last = (i == len(turns) - 1)
        if explicit[i]:
            searches = explicit[i]
        elif search_budget > 0 and (not is_last or len(turns) == 1):
            searches = [rec.get("question", "query")]
        else:
            searches = []
        search_budget -= len(searches)
        plan.append((min(toks, 1024), searches))
    if not plan:
        plan = [(64, [])]
    return plan


# --------------------------------------------------------------------------
# vLLM async engine + KV-cache-hit stat logger
# --------------------------------------------------------------------------
# module-level accumulator so the custom logger (constructed by vLLM per engine)
# can publish prefix-cache stats back to the driver.
KV_STATS = {"queries": 0, "hits": 0}


def make_kv_logger():
    from vllm.v1.metrics.loggers import StatLoggerBase

    class KVHitLogger(StatLoggerBase):
        """Accumulates token-level prefix-cache hit stats across scheduler steps.
        KV hit rate = sum(hits)/sum(queries) over the run."""
        def __init__(self, vllm_config, engine_index=0):
            self._last_reset_seen = False

        def record(self, scheduler_stats, iteration_stats):
            if scheduler_stats is None:
                return
            pcs = getattr(scheduler_stats, "prefix_cache_stats", None)
            if pcs is not None:
                KV_STATS["queries"] += int(pcs.queries)
                KV_STATS["hits"] += int(pcs.hits)

        def log(self, *a, **k):
            pass

        def log_engine_initialized(self, *a, **k):
            pass

    return lambda vllm_config, idx=0: KVHitLogger(vllm_config, idx)


def make_engine(args):
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    ea = AsyncEngineArgs(
        model=args.llm,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        disable_log_requests=True,
        dtype="float16",
        enable_prefix_caching=True,   # KV reuse across multi-turn re-sends
    )
    return AsyncLLMEngine.from_engine_args(ea, stat_loggers=[make_kv_logger()])


async def run_request(engine, retriever, rec, ridx, dispatcher, loop, results, slo_ms, max_model_len, no_thinking=True):
    """One request: multi-turn generate + retrieval, oracle docs injected."""
    from vllm import SamplingParams
    plan = turn_plan(rec, no_thinking=no_thinking)
    # Qwen3 /no_think switch disables the reasoning block (matches HedraRAG replay).
    sys_hint = " /no_think" if no_thinking else ""
    prompt = f"Question: {rec['question']}{sys_hint}\n"
    t_start = time.time()
    ttft = None
    retr_wall_ms = 0.0      # wall incl. queue wait behind executor / GPU lock
    retr_compute_ms = 0.0   # pure retriever compute (quantizer + scan), no queueing
    n_search = 0

    for turn_i, (toks, searches) in enumerate(plan):
        # keep prompt + max_tokens within the model window. ~3.5 chars/token; reserve
        # `toks` for generation, cap prompt context to the rest, truncating the middle.
        max_ctx_chars = int(max(1000, (max_model_len - toks - 256)) * 3.2)
        if len(prompt) > max_ctx_chars:
            head = prompt[: int(max_ctx_chars * 0.3)]
            tail = prompt[-int(max_ctx_chars * 0.7):]
            prompt = head + "\n...\n" + tail
        sp = SamplingParams(temperature=0.0, max_tokens=toks, min_tokens=min(toks, 8))
        rid = f"r{ridx}-t{turn_i}"
        first = True
        async for out in engine.generate(prompt, sp, rid):
            if first and ttft is None:
                ttft = time.time() - t_start
                first = False
        gen_text = out.outputs[0].text if out.outputs else ""
        prompt += gen_text

        # retrieval on <search> turns
        if searches:
            t_r = time.time()
            if dispatcher is not None:
                # cross-request micro-batched dispatch (paper sec IV-B-2)
                comps = await asyncio.gather(*[dispatcher.search(s) for s in searches])
                retr_compute_ms += max((c for c in comps if c is not None), default=0.0)
            else:
                holder = {}
                def do_search():
                    retriever.search(searches, retriever.topk)
                    # pure compute reported by the engine (no queue wait)
                    holder["c"] = getattr(retriever, "last_scan_ms", 0.0) + \
                                  getattr(retriever, "last_quantizer_ms", 0.0)
                await loop.run_in_executor(None, do_search)
                retr_compute_ms += holder.get("c", 0.0)
            retr_wall_ms += (time.time() - t_r) * 1000
            n_search += len(searches)
            # inject ORACLE info from the trace (retriever result discarded)
            infos = INFO_RE.findall(rec.get("reasoning_trace", "") or "")
            inj = infos[turn_i] if turn_i < len(infos) else ""
            prompt += f"\n<information>{inj}</information>\n"

    e2e = time.time() - t_start
    ttft_ms = (ttft or e2e) * 1000
    results.append(dict(
        ridx=ridx, ttft_ms=ttft_ms, e2e_ms=e2e * 1000,
        retrieval_wall_ms=retr_wall_ms, retrieval_compute_ms=retr_compute_ms,
        retrieval_queue_ms=max(0.0, retr_wall_ms - retr_compute_ms),
        n_search=n_search, n_turns=len(plan),
        # search-stage SLO: judged against pure retrieval COMPUTE latency (paper's
        # SLO_search targets retrieval); also keep a separate TTFT-SLO view.
        search_slo_ok=bool((retr_compute_ms / max(n_search, 1)) <= slo_ms) if n_search else True,
        ttft_slo_ok=bool(ttft_ms <= slo_ms),
    ))


async def replay_rps(engine, retriever, recs, rps, args):
    loop = asyncio.get_event_loop()
    results = []
    tasks = []
    dispatcher = None
    if args.dispatcher:
        dispatcher = RetrievalDispatcher(retriever, loop,
                                         max_batch=args.disp_max_batch,
                                         window_s=args.disp_window_ms / 1000.0)
    rng = np.random.default_rng(args.seed)
    # build poisson arrival schedule up front (override recorded offsets)
    gaps = rng.exponential(1.0 / rps, size=len(recs)) if rps > 0 else np.zeros(len(recs))
    arrivals = np.cumsum(gaps)
    t0 = time.time()
    for ridx, rec in enumerate(recs):
        now = time.time() - t0
        wait = arrivals[ridx] - now
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(
            run_request(engine, retriever, rec, ridx, dispatcher, loop, results, args.search_slo_ms, args.max_model_len, args.no_thinking)))
    await asyncio.gather(*tasks)
    wall = time.time() - t0
    return results, wall


def summarize(results, wall, rps, kv_queries=0, kv_hits=0):
    ttft = np.array([r["ttft_ms"] for r in results])
    e2e = np.array([r["e2e_ms"] for r in results])
    rc = np.array([r["retrieval_compute_ms"] for r in results])
    rq = np.array([r["retrieval_queue_ms"] for r in results])
    def pct(a, p): return float(np.percentile(a, p)) if len(a) else 0.0
    return dict(
        rps=rps, n=len(results), wall_s=wall,
        # end-to-end throughput (completed requests / wall clock)
        e2e_throughput=len(results) / wall,
        # e2e latency percentiles (service metric)
        e2e_avg=float(e2e.mean()), e2e_p50=pct(e2e, 50),
        e2e_p90=pct(e2e, 90), e2e_p99=pct(e2e, 99),
        # TTFT percentiles
        ttft_avg=float(ttft.mean()), ttft_p50=pct(ttft, 50),
        ttft_p90=pct(ttft, 90), ttft_p99=pct(ttft, 99),
        # retrieval: separated compute vs queue-wait (fixes the inflated metric)
        retr_compute_avg=float(rc.mean()), retr_compute_p90=pct(rc, 90),
        retr_queue_avg=float(rq.mean()), retr_queue_p90=pct(rq, 90),
        # SLO attainment: search-stage (retrieval compute) and TTFT views
        search_slo_attainment=float(np.mean([r["search_slo_ok"] for r in results])),
        ttft_slo_attainment=float(np.mean([r["ttft_slo_ok"] for r in results])),
        # KV cache hit rate (token-level prefix cache reuse)
        kv_cache_hit_rate=(kv_hits / kv_queries) if kv_queries else 0.0,
        kv_queries=int(kv_queries), kv_hits=int(kv_hits),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", choices=["cpu", "vlite"], default="vlite")
    ap.add_argument("--index", default=os.environ.get("SR1_INDEX"))
    ap.add_argument("--trace", default=os.environ.get("SR1_TRACE"))
    ap.add_argument("--e5", default=os.environ.get("SR1_E5"))
    ap.add_argument("--llm", default=os.environ.get("SR1_LLM"))
    ap.add_argument("--part_dir", default="database/sr1/4gpus_hi")
    ap.add_argument("--out_dir", default="runs/sr1_run")
    ap.add_argument("--rps_list", default="2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nprobe", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_mem_util", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--search_slo_ms", type=float, default=400.0)
    ap.add_argument("--enable_swap", action="store_true")
    ap.add_argument("--cpu_in_ram", action="store_true")
    ap.add_argument("--dispatcher", action="store_true",
                    help="enable cross-request micro-batched dynamic dispatcher (sec IV-B-2)")
    ap.add_argument("--disp_max_batch", type=int, default=16)
    ap.add_argument("--disp_window_ms", type=float, default=5.0)
    ap.add_argument("--no_thinking", dest="no_thinking", action="store_true", default=True,
                    help="strip <think> and gen only search/answer (HedraRAG default, ~11x fewer tokens)")
    ap.add_argument("--thinking", dest="no_thinking", action="store_false",
                    help="full reasoning replay (uses step_completion_tokens)")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    recs = load_trace(args.trace, args.limit)
    log(f"loaded {len(recs)} trace records")

    log(f"building retriever={args.retriever}")
    retriever = build_retriever(args)
    # warm the encoder + one search
    retriever.search([recs[0]["question"]], args.topk)
    log("retriever warm")

    log("initializing vLLM AsyncLLMEngine (this can take a minute)")
    engine = make_engine(args)

    rps_values = [float(x) for x in args.rps_list.split(",")]

    async def run_all():
        # all rps levels share ONE event loop + engine (AsyncLLMEngine binds its
        # background loop to the loop it was first awaited in; a fresh asyncio.run
        # per level kills the engine with EngineDeadError).
        all_summaries = []
        for rps in rps_values:
            log(f"=== replay rps={rps} ===")
            KV_STATS["queries"] = 0    # reset per-run KV accounting
            KV_STATS["hits"] = 0
            results, wall = await replay_rps(engine, retriever, recs, rps, args)
            s = summarize(results, wall, rps, KV_STATS["queries"], KV_STATS["hits"])
            all_summaries.append(s)
            log(f"rps={rps}: e2e_tput={s['e2e_throughput']:.2f} "
                f"e2e[p50/p90/p99]={s['e2e_p50']:.0f}/{s['e2e_p90']:.0f}/{s['e2e_p99']:.0f}ms "
                f"retr_compute_avg={s['retr_compute_avg']:.0f}ms(queue={s['retr_queue_avg']:.0f}) "
                f"kv_hit={s['kv_cache_hit_rate']:.3f} "
                f"search_slo={s['search_slo_attainment']:.3f} ttft_slo={s['ttft_slo_attainment']:.3f}")
            with open(out / f"records_rps{rps}.jsonl", "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
        return all_summaries

    all_summaries = asyncio.run(run_all())

    with open(out / "summary.json", "w") as f:
        json.dump(dict(retriever=args.retriever, nprobe=args.nprobe,
                       part_dir=args.part_dir, dispatcher=args.dispatcher,
                       no_thinking=args.no_thinking,
                       summaries=all_summaries), f, indent=2)
    log(f"DONE -> {out}/summary.json")


if __name__ == "__main__":
    main()
