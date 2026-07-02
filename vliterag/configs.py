import os
import re
import torch
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from configs.loader import load_model, load_index

PRJ_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class vLiteConfigs:
    """Configuration class for VectorLiteRAG."""
    
    model: str = "llama8b"           
    index: str = "wikiall"

    gpu_type: str = None
    gpu_util: float = 0.95
    num_gpus: int = 8

    tp_size: int = 0
    input_len: int = 1024
    output_len: int = 256
    
    search_mode: str | None = None       # all | cpu | all-gpu | ded-gpu | vlite | hedrarag | None 
    search_slo: int = -1                 # ms (vlite only)
    search_topk: int = 25
    search_nprobe: int = 2048
    ann_workers: int = 1

    dispatcher: bool = True
    sweep: bool = False            
    is_profiling: bool = False
    is_plotting: bool = False

    arrival_rate: float = 0.0
    running_time: int = 120             # for serving job this represents of duration of request stream, for profiling job number of repetition 

    database_dir: Path | str = (PRJ_ROOT / 'database').resolve()
    result_dir: Path | str = field(init=False)

    eager_mode: bool = False
    
    model_cfg: dict = field(init=False)
    index_cfg: dict = field(init=False)
    llm_workers: int = field(init=False)
    total_requests: int = field(init=False)
    warmup_requests: int = field(init=False)
    file_name: str = field(init=False)
    file_tag: str = 'main'
        
    def __post_init__(self):
        self.model_cfg = load_model(self.model)
        self.index_cfg = load_index()[self.index]

        if self.tp_size == 0:
            self.tp_size = self.model_cfg["tp_size"]

        if not self.gpu_type:
            if torch.cuda.is_available():
                match = re.search(r'\b(H200|H100|A100|L40S|RTX 4090|RTX 3090)\b', torch.cuda.get_device_name(0))
                if match:
                    self.gpu_type = match.group(1)
                else:
                    raise ValueError(f"GPU type not specified")
            else:
                raise ValueError(f"GPU type not specified")

        if self.gpu_type == 'L40S' and self.model != 'llama8b':
            self.eager_mode = True

        self.resolve_mode_dependencies()

        if self.is_profiling:
            self.arrival_rate = 0
            self.set_profiling_requests()
        else:
            if self.arrival_rate == 0:
                max_rate = self.get_tput_ceiling()
                self.arrival_rate = self.get_arrival_rates(max_rate)[0]
            self.set_request_counts()

        self.set_result_paths()
        self.log_config()

        os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    def get_tput_ceiling(self, in_len=None, out_len=None, ngpu=None):
        in_len = in_len or self.input_len
        out_len = out_len or self.output_len
        ngpu = ngpu or self.num_gpus

        ngpu_key = f"ngpu={ngpu}"
        
        entries = self.model_cfg["tput_ceiling"].get(self.gpu_type, [])
        for e in entries:
            if e["input"] == in_len and e["output"] == out_len:
                return e["values"][ngpu_key]

        return -1

    def get_model_slo(self, in_len=None, out_len=None, ngpu=None):
        in_len = in_len or self.input_len
        out_len = out_len or self.output_len
        ngpu = ngpu or self.num_gpus
        ngpu_key = f"ngpu={ngpu}"

        entries = self.model_cfg["slo"].get(self.gpu_type, [])
        for e in entries:
            if e["input"] == in_len and e["output"] == out_len:
                return e["values"][ngpu_key]

        raise ValueError(
            f"No SLO entry for device={self.gpu_type}, input={in_len}, output={out_len}, ngpu={ngpu}"
        )

    def get_arrival_rates(self, max_rate):
        fracs = [
            8/16, 10/16, 12/16, 13/16,
            27/32, 14/16, 29/32, 15/16,
            31/32, 1.0, 33/32,
        ]

        if max_rate >= 16:
            return [round(max_rate * f) for f in fracs]
        rates = [max_rate * f for f in fracs]  
        if 17 < max_rate < 18:
            rates += [19, 20]
        return rates
    
    def resolve_mode_dependencies(self):
        if self.search_mode == "ded-gpu":
            # hard coded ann_workers for ded-gpu mode
            if self.index == 'orcas2k' and self.gpu_type == 'L40S':
                self.ann_workers = 4  
            else:
                self.ann_workers = self.tp_size
            self.llm_workers = (self.num_gpus - self.ann_workers) // self.tp_size
        else:
            self.llm_workers = self.num_gpus // self.tp_size
            
        if self.search_mode != 'vlite':
            self.dispatcher = False
            if not self.is_profiling:
                self.search_slo = -1
        elif self.search_slo < 0:
            self.search_slo = self.index_cfg['slo']
            
    def set_profiling_requests(self):
        if self.search_mode is None:
            self.total_requests = self.running_time
            self.warmup_requests = 0
        else:
            self.total_requests = int((32 * 33) // 2) * self.running_time
            self.warmup_requests = self.llm_workers

    def set_request_counts(self):
        self.total_requests = int(self.running_time * self.arrival_rate)
        self.warmup_requests = self.llm_workers

    def set_result_paths(self):
        parts = {
            "slo": f"S{self.search_slo}" if self.search_mode == "vlite" else "S0",
            "input_len": f"I{self.input_len}",
            "output_len": f"O{self.output_len}",
            "arrival_rate": f"R{int(self.arrival_rate)}" if float(self.arrival_rate).is_integer()
                            else f"R{self.arrival_rate:.2f}",
            "num_workers": f"N{self.llm_workers}",
            "gpu_util": f"U{int(self.gpu_util * 100)}",
            "nprobe": f"P{self.search_nprobe}",
        }

        file_name = "_".join(parts.values())
        if not self.dispatcher:
            file_name += "_Dx"

        self.file_name = file_name

        base = (PRJ_ROOT / 'results').resolve()
        base = base / self.index / self.model / f"{self.num_gpus}gpus"

        if self.search_mode:
            base = base / self.search_mode

        self.result_dir = base
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        (self.result_dir / 'summary').mkdir(parents=True, exist_ok=True)
        (self.result_dir / 'raw').mkdir(parents=True, exist_ok=True)

    def update_and_sweep(self, new_rate=None):
        if not self.sweep:
            return False
        
        if self.is_profiling:
            return False
        
        updated = False
        if new_rate is not None:
            self.arrival_rate = new_rate
            updated = True
        else:
            for rate in self.get_arrival_rates(self.get_tput_ceiling()):
                if rate > self.arrival_rate:
                    self.arrival_rate = rate
                    updated = True
                    break
        
        if updated:
            self.total_request = int(self.running_time * self.arrival_rate)
            self.set_result_paths()
            self.log_config()
        
        return updated      

    def log_config(self):
        if self.is_plotting or self.is_profiling:
            return
        
        print(f"[VLITE] model: {self.model_cfg['name']}, tp={self.tp_size}, in/out={self.input_len}/{self.output_len}")
        if self.search_mode:
            print(f"[VLITE] index: {self.index}, search_mode={self.search_mode}, slo={self.search_slo}")
        print(f"[VLITE] gpus: {self.gpu_type} x{self.num_gpus}, requests={self.total_requests}, arrival={self.arrival_rate:.2f}")