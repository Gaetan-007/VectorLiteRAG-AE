#!/usr/bin/env python3
"""Aggregate the isolated per-point runs under runs/vlite_full/rps*/ into one
markdown + CSV summary. Each point was collected in its own vLLM process so points
do not interfere."""
import json
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "runs" / "vlite_full"

COLS = [
    ("rps", "rps"),
    ("e2e_throughput", "e2e tput"),
    ("e2e_avg", "e2e avg"),
    ("e2e_p50", "e2e p50"),
    ("e2e_p90", "e2e p90"),
    ("e2e_p99", "e2e p99"),
    ("kv_cache_hit_rate", "KV hit"),
    ("retr_compute_avg", "retr compute"),
    ("retr_queue_avg", "retr queue"),
    ("search_slo_attainment", "search SLO"),
    ("ttft_slo_attainment", "TTFT SLO"),
]


def load_points():
    pts = []
    for sp in glob.glob(str(OUT_ROOT / "rps*" / "summary.json")):
        d = json.load(open(sp))
        s = d["summaries"][0]
        s["_meta"] = {k: d.get(k) for k in ("retriever", "nprobe", "dispatcher",
                                            "no_thinking", "part_dir")}
        pts.append(s)
    # sort: offline burst (-1) last, else ascending rps
    pts.sort(key=lambda s: (s["rps"] < 0, s["rps"]))
    return pts


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def main():
    pts = load_points()
    m = pts[0]["_meta"] if pts else {}
    header = (
        "# Single 4090 — Full-opt vlite, isolated 13-point Search-R1 sweep\n"
        f"Config: retriever=vlite, no_thinking={m.get('no_thinking')}, "
        f"dispatcher={m.get('dispatcher')}, nprobe={m.get('nprobe')}, "
        f"part_dir={m.get('part_dir')}, prefix-cache ON. Each rps point = its own "
        "vLLM process on a dedicated GPU (points do not interfere). limit=200, "
        "records-hotpotqa, seed=2025.\n\n"
    )
    # markdown table
    head = "| " + " | ".join(lbl for _, lbl in COLS) + " |"
    sep = "|" + "|".join("---:" for _ in COLS) + "|"
    rows = []
    for s in pts:
        rows.append("| " + " | ".join(fmt(s.get(k, "")) for k, _ in COLS) + " |")
    md = header + head + "\n" + sep + "\n" + "\n".join(rows) + "\n"
    (OUT_ROOT / "RESULTS.md").write_text(md)

    # csv
    csv = ",".join(k for k, _ in COLS) + "\n"
    for s in pts:
        csv += ",".join(str(s.get(k, "")) for k, _ in COLS) + "\n"
    (OUT_ROOT / "RESULTS.csv").write_text(csv)

    print(md)
    print(f"wrote {OUT_ROOT/'RESULTS.md'} and RESULTS.csv ({len(pts)} points)")


if __name__ == "__main__":
    main()
