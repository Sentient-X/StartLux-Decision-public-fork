"""Uniformly average the weights of independently seeded fine-tuning runs, as the release models were made
(technical report §2.3.2: "each released Instant model is obtained by uniformly averaging the weights of three
independently trained runs ... differing only in their random seeds"; temperatures are refitted afterwards).

    python -m sentient.average --out runs/avg runs/seed0/merged runs/seed1/merged runs/seed2/merged
    python v1/finetune/calibrate.py --model runs/avg --dev dev.jsonl

Every input must be a merged folder of the same base (finetune_lora.py writes <out>/merged). Tensors are averaged in
float32 and written in each tensor's own dtype; the other files of the folder are copied from the first input.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    runs, out = [Path(r) for r in a.runs], Path(a.out)
    if len(runs) < 2:
        ap.error("average at least two runs")
    shards = sorted(p.name for p in runs[0].glob("*.safetensors"))
    for r in runs[1:]:
        if sorted(p.name for p in r.glob("*.safetensors")) != shards:
            raise SystemExit(f"{r}: different weight files than {runs[0]}")
    out.mkdir(parents=True, exist_ok=True)
    for f in runs[0].iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy(f, out / f.name)
    for name in shards:
        handles = [safe_open(str(r / name), "pt") for r in runs]
        keys = list(handles[0].keys())
        merged = {}
        for k in keys:
            first = handles[0].get_tensor(k)
            acc = first.float().clone()
            for h in handles[1:]:
                acc += h.get_tensor(k).float()
            merged[k] = (acc / len(runs)).to(first.dtype)
        save_file(merged, str(out / name), metadata={"format": "pt"})
        print(name, len(keys), "tensors", flush=True)
    json.dump({"averaged": [str(r) for r in runs]}, open(out / "sentient_average.json", "w"), indent=1)
    print("averaged", len(runs), "runs into", out, "- refit temperatures with v1/finetune/calibrate.py")


if __name__ == "__main__":
    main()
