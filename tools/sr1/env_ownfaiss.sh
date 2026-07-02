#!/usr/bin/env bash
# Environment using THIS REPO's own custom-built faiss (not HedraRAG's).
# Usage: source tools/sr1/env_ownfaiss.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SP="$HERE/.venv/lib/python3.10/site-packages"
CUDAENV="$HERE/.cudaenv"

# nvidia wheel libs (torch runtime) + MKL symlink + the CUDA-toolkit runtime libs
# that the custom faiss .so links against (cudart/cublas 12.9, MKL from .cudaenv)
NVLIBS="$(find "$SP/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="$HERE/.venv/lib:$CUDAENV/lib:$CUDAENV/targets/x86_64-linux/lib:${NVLIBS}${LD_LIBRARY_PATH:-}"

# THIS repo's custom faiss (has set_gpu_index / register_callback / cacheflow)
export VLITE_FAISS_PY="$HERE/faiss/build/faiss/python/build/lib"
export PYTHONPATH="$VLITE_FAISS_PY:$HERE:${PYTHONPATH:-}"

export VLITE_PY="$HERE/.venv/bin/python"
export VLITE_ROOT="$HERE"

export SR1_INDEX="/mnt/public/Agent_index/wiki-25-corpus/e5_IVF4096,Flat.index"
export SR1_CORPUS="/mnt/public/Agent_index/wiki-25-corpus/wiki-25-chunk-256.jsonl"
export SR1_E5="/mnt/public/multilingual-e5-large-instruct"
export SR1_LLM="/mnt/public/Qwen3-4B"
export SR1_TRACE="/mnt/public/wangzehao/AGInf-Mem/data/search-r1/records-hotpotqa.jsonl"
