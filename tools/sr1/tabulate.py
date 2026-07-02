#!/usr/bin/env python3
"""Tabulate single-4090 Search-R1 replay results: e2e throughput, e2e latency
percentiles (avg/p50/p90/p99), KV-cache hit rate, retrieval compute vs queue,
and search/TTFT SLO attainment."""
import json

RUNS = {
    "CPU-only":   "runs/1gpu_cpu/summary.json",
    "vlite":      "runs/1gpu_vlite/summary.json",
    "vlite+disp": "runs/1gpu_vlite_disp/summary.json",
}


def main():
    data = {}
    for name, path in RUNS.items():
        try:
            d = json.load(open(path))
        except FileNotFoundError:
            continue
        for s in d["summaries"]:
            data.setdefault(s["rps"], {})[name] = s

    lines = [
        "# Single 4090 — Search-R1 replay (VectorLiteRAG reproduction)",
        "Qwen3-4B tp=1 co-located with faiss on ONE RTX 4090 24GB, prefix-caching ON. "
        "HedraRAG-aligned: nprobe=512, topk=3, seed=2025, records-hotpotqa.",
        "",
        "| rps | config | e2e tput | e2e avg | e2e p50 | e2e p90 | e2e p99 | KV hit | "
        "retr compute (ms) | retr queue (ms) | search SLO | TTFT SLO |",
        "|----:|:-------|---------:|--------:|--------:|--------:|--------:|-------:|"
        "------------------:|----------------:|-----------:|---------:|",
    ]
    for rps in sorted(data):
        for name in ["CPU-only", "vlite", "vlite+disp"]:
            s = data[rps].get(name)
            if not s:
                continue
            lines.append(
                f"| {rps} | {name} | {s['e2e_throughput']:.2f} | {s['e2e_avg']:.0f} | "
                f"{s['e2e_p50']:.0f} | {s['e2e_p90']:.0f} | {s['e2e_p99']:.0f} | "
                f"{s['kv_cache_hit_rate']:.3f} | {s['retr_compute_avg']:.0f} | "
                f"{s['retr_queue_avg']:.0f} | {s['search_slo_attainment']:.2f} | "
                f"{s['ttft_slo_attainment']:.2f} |")
    out = "\n".join(lines) + "\n"
    open("runs/RESULTS_1gpu.md", "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
