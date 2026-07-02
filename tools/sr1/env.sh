#!/usr/bin/env bash
# Repo-local environment for VectorLiteRAG Search-R1 replay.
# Usage:  source tools/sr1/env.sh   (then run .venv/bin/python ...)
# Created by the SR1 reproduction work. Uses uv venv + HedraRAG's prebuilt GPU faiss.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SP="$HERE/.venv/lib/python3.10/site-packages"

# CUDA runtime libs bundled inside the torch/nvidia wheels + MKL (libmkl_rt.so.2 symlink)
NVLIBS="$(find "$SP/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="$HERE/.venv/lib:${NVLIBS}${LD_LIBRARY_PATH:-}"

# Prebuilt modified-FAISS (GPU + search_preassigned). NOTE: imported read-only as a
# library path; we do NOT modify HedraRAG. If you build this repo's own faiss, point here instead.
export VLITE_FAISS_PY="${VLITE_FAISS_PY:-/mnt/public/wangzehao/HedraRAG_AE/faiss/build/faiss/python}"
export PYTHONPATH="$VLITE_FAISS_PY:$HERE:${PYTHONPATH:-}"

export VLITE_PY="$HERE/.venv/bin/python"
export VLITE_ROOT="$HERE"

# Search-R1 assets (read-only)
export SR1_INDEX="/mnt/public/Agent_index/wiki-25-corpus/e5_IVF4096,Flat.index"
export SR1_CORPUS="/mnt/public/Agent_index/wiki-25-corpus/wiki-25-chunk-256.jsonl"
export SR1_E5="/mnt/public/multilingual-e5-large-instruct"
export SR1_LLM="/mnt/public/Qwen3-4B"
export SR1_TRACE="/mnt/public/wangzehao/AGInf-Mem/data/search-r1/records-hotpotqa.jsonl"
