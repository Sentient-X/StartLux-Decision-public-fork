"""Put Sentient Decision's work on the sx platform through Waddle (Console /training), without touching the trainer.

    python -m sentient.track follow --log runs/pilot.log --run-id sentient-pilot --name "pilot 4B" --config mix/manifest.json
    python -m sentient.track bench --score preds/score.json --run-id sentient-bench-pilot --name "bench: pilot test set"
    python -m sentient.track data --manifest mix/v1/manifest.json --run-id sentient-data-v1 --name "mix v1"

`follow` reads finetune_lora.py's log from its start (so an already running fine-tune can be attached) and keeps
reading until the run writes its merged model or fails: step lines become train/loss and train/lr, dev evaluations
become dev/loss and dev/accuracy, other lines become log lines. Its read position is saved beside the log, so a
restarted follower never logs a point twice. With SX_WADDLE_URL, SX_PROJECT_ID and SX_API_KEY set, Waddle streams the
run to the platform; without them it records locally and `waddle sync` sends it later. Waddle never blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

GROUP = "sentient-decision"
_PROGRESS = re.compile(r"(Loading weights|Fetching|it/s\]|%\|)")


def run_id(name: str) -> str:
    """Waddle's hosted run ids are 32 hex digits; a stable one per logical name keeps resume on the same run."""
    import uuid
    return uuid.uuid5(uuid.NAMESPACE_URL, f"sentient-decision/{name}").hex


def _init(a, run_type, extra_config=None):
    import waddle
    config = {}
    for path in a.config or []:
        config[Path(path).stem] = json.loads(Path(path).read_text())
    config.update(extra_config or {})
    return waddle.init(run_id=run_id(a.run_id), resume=True, display_name=a.name, description=a.description,
                       group_name=GROUP, run_type=waddle.RunType(run_type), config=config, system_metrics=a.system)


def follow(a):
    import waddle
    log = Path(a.log)
    pos_file = log.with_suffix(log.suffix + ".waddle-pos")
    state = json.loads(pos_file.read_text()) if pos_file.exists() else {"offset": 0, "step": 0}
    _init(a, "training")
    while not log.exists():
        time.sleep(a.poll)
    finished = False
    with open(log, "rb") as f:
        f.seek(state["offset"])
        buf = b""
        while not finished:
            chunk = f.read(1 << 16)
            if not chunk:
                pos_file.write_text(json.dumps(state))
                time.sleep(a.poll)
                continue
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)
            for raw in lines:
                line = raw.decode("utf-8", "replace").strip()
                if not line or _PROGRESS.search(line):
                    continue
                if line.startswith("{"):
                    try:
                        doc = json.loads(line)
                    except ValueError:
                        waddle.log_line(line)
                        continue
                    if "step" in doc and "loss" in doc:
                        state["step"] = int(doc["step"])
                        waddle.log({"train/loss": doc["loss"], "train/lr": doc["lr"],
                                    "train/elapsed_s": doc.get("elapsed_s", 0)}, step=state["step"])
                    for key in ("before", "done"):
                        if isinstance(doc.get(key), dict):
                            dev = {f"dev/{k.removeprefix('dev_')}": v for k, v in doc[key].items()
                                   if k.startswith("dev_")}
                            if dev:
                                waddle.log(dev, step=state["step"])
                    if "epoch" in doc:
                        dev = {f"dev/{k.removeprefix('dev_')}": v for k, v in doc.items() if k.startswith("dev_")}
                        waddle.log({"train/epoch": doc["epoch"], **dev}, step=state["step"])
                    if "optimizer_steps" in doc:
                        waddle.log_line(f"plan: {doc['optimizer_steps']} optimizer steps, "
                                        f"{doc.get('micro_batches_per_epoch')} micro-batches per epoch")
                    continue
                level = "error" if ("Traceback" in line or "Error" in line) else "info"
                waddle.log_line(line[:2000], level=level)
                if line.startswith("merged model written") or (a.stop_on_error and level == "error"):
                    finished = True
            state["offset"] = f.tell() - len(buf)
    pos_file.write_text(json.dumps(state))
    waddle.finish()


def bench(a):
    import waddle
    report = json.loads(Path(a.score).read_text())
    for path, groups in report.items():
        model = Path(path).stem
        _init(argparse.Namespace(**{**vars(a), "run_id": f"{a.run_id}-{model}", "name": f"{a.name}: {model}"}),
              "evaluation", {"model": model, "cost_usd": groups.get("cost_usd")})
        for g, m in groups.items():
            if isinstance(m, dict):
                waddle.log({f"{g}/{k}": v for k, v in m.items() if isinstance(v, (int, float))}, step=0)
        waddle.finish()


def data(a):
    import waddle
    manifest = json.loads(Path(a.manifest).read_text())
    _init(a, "data", {"manifest": manifest})
    flat = {}
    for src, splits in manifest.get("sources", {}).items():
        for split, n in splits.items():
            flat[f"records/{src}/{split}"] = n
    for split in ("train", "dev", "test"):
        if split in manifest:
            flat[f"questions/{split}"] = manifest[split]["questions"]
    waddle.log(flat, step=0)
    waddle.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("follow", "bench", "data"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--description")
        p.add_argument("--config", action="append", help="JSON files recorded as run config")
        p.add_argument("--system", action="store_true", help="also record CPU/GPU/memory (follow only)")
        if name == "follow":
            p.add_argument("--log", required=True)
            p.add_argument("--poll", type=float, default=10.0)
            p.add_argument("--stop-on-error", action="store_true")
        if name == "bench":
            p.add_argument("--score", required=True)
        if name == "data":
            p.add_argument("--manifest", required=True)
    a = ap.parse_args()
    if os.environ.get("SX_WADDLE_URL") and not os.environ.get("SX_PROJECT_ID"):
        ap.error("SX_WADDLE_URL is set without SX_PROJECT_ID")
    {"follow": follow, "bench": bench, "data": data}[a.cmd](a)


if __name__ == "__main__":
    main()
