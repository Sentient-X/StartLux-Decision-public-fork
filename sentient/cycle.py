"""Keep training as data arrives: rebuild the sources, mix, and when the training set has grown enough, fine-tune a
fresh run from the base model, calibrate it, and bench it, reporting every step to Waddle (Console /training).

    python -m sentient.cycle --base $BASE --raw ~/sd-data/raw --work ~/sd-data --runs ~/sd-runs \\
        --source fineart --source failtime --source roboarena:30000 --test-source robometer_ood

Each cycle is one version (v1, v2, ...): <work>/built/<source>, <work>/mix/vN, <runs>/vN. A version trains only if its
training set has at least --grow times the questions of the last trained one; otherwise the cycle waits and looks
again. Each version starts from the base model (the report's Stage 2 trains adapters from the same checkpoint), so
versions are comparable and a bad one costs nothing later. Nothing is retried blindly: a failed step stops the loop.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(cmd: list, log: Path, background: bool = False):
    log.parent.mkdir(parents=True, exist_ok=True)
    f = open(log, "ab")
    print(time.strftime("%H:%M:%S"), " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if background:
        return p
    if p.wait() != 0:
        raise SystemExit(f"step failed ({p.returncode}): {' '.join(map(str, cmd))}; see {log}")
    return p


def version_numbers(mixes: Path) -> list[int]:
    return sorted(int(p.name[1:]) for p in mixes.glob("v*") if p.name[1:].isdigit())


def cycle(a, n: int, last_trained: int) -> int:
    work, runs = Path(a.work), Path(a.runs)
    logs = runs / f"v{n}" / "logs"
    for spec in a.source + a.test_source:
        name = spec.split(":")[0]
        run([PY, "-m", "sentient.data.build", "--source", name, "--raw", a.raw, "--out", work / "built" / name,
             "--windows", a.windows, "--workers", a.workers], logs / f"build-{name}.log")
    mix = work / "mix" / f"v{n}"
    run([PY, "-m", "sentient.data.mix", "--built", work / "built", "--out", mix, "--seed", f"v{n}",
         "--retention", a.retention] + sum((["--source", s] for s in a.source), []), logs / "mix.log")
    for name in a.test_source:             # test-only sources join the bench set, never train
        with open(mix / "test.jsonl", "a") as dst:
            src = work / "built" / name / "test.jsonl"
            rel = Path("..") / ".." / "built" / name
            for line in list(open(src))[:400]:
                rec = json.loads(line)
                rec["images"] = [str(rel / p) for p in rec.get("images") or []]
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = json.loads((mix / "manifest.json").read_text())
    questions = manifest["train"]["questions"]
    run([PY, "-m", "sentient.track", "data", "--manifest", mix / "manifest.json", "--run-id", f"sentient-data-v{n}",
         "--name", f"Sentient Decision data v{n}: {questions} training questions"], logs / "track-data.log")
    if last_trained and questions < a.grow * last_trained:
        print(f"v{n}: {questions} questions, under {a.grow}x the last trained ({last_trained}); waiting", flush=True)
        return last_trained
    out = runs / f"v{n}"
    train_log = out / "train.log"
    follower = run([PY, "-m", "sentient.track", "follow", "--log", train_log, "--run-id", f"sentient-train-v{n}",
                    "--name", f"Sentient Decision v{n} 4B ({questions} questions)", "--config", mix / "manifest.json",
                    "--system", "--stop-on-error"], logs / "track-train.log", background=True)
    try:
        run([PY, ROOT / "v1/finetune/finetune_lora.py", "--model", a.base, "--train", mix / "train.jsonl",
             "--dev", mix / "dev.jsonl", "--out", out, "--seed", "0", "--epochs", a.epochs], train_log)
    finally:
        follower.wait(timeout=900)
    model = out / "merged"
    run([PY, ROOT / "v1/finetune/calibrate.py", "--model", model, "--dev", mix / "dev.jsonl"], logs / "calibrate.log")
    preds = out / "preds"
    run([PY, "-m", "sentient.bench", "predict", "--model", model, "--data", mix / "test.jsonl",
         "--out", preds / f"sentient-v{n}.jsonl"], logs / "bench.log")
    run([PY, "-m", "sentient.bench", "predict", "--model", a.base, "--data", mix / "test.jsonl",
         "--out", preds / "base.jsonl"], logs / "bench-base.log")
    score = preds / "score.json"
    run([PY, "-m", "sentient.bench", "score", preds / f"sentient-v{n}.jsonl", preds / "base.jsonl",
         "--data", mix / "test.jsonl", "--train", mix / "train.jsonl", "--json", score], logs / "score.log")
    run([PY, "-m", "sentient.track", "bench", "--score", score, "--run-id", f"sentient-bench-v{n}",
         "--name", f"Sentient Decision bench v{n}"], logs / "track-bench.log")
    if a.rrb:                              # RoboRewardBench as FLORA reports it; base predictions are made once and kept
        rrb = Path(a.rrb)
        base_preds = rrb.parent / "preds" / "rrb-base.jsonl"
        if not base_preds.exists():
            run([PY, "-m", "sentient.bench", "predict", "--model", a.base, "--data", rrb, "--out", base_preds],
                logs / "rrb-base.log")
        mine = preds / f"rrb-sentient-v{n}.jsonl"
        run([PY, "-m", "sentient.bench", "predict", "--model", model, "--data", rrb, "--out", mine], logs / "rrb.log")
        rrb_score = preds / "rrb-score.json"
        run([PY, "-m", "sentient.roborewardbench", "score", mine, base_preds, "--data", rrb, "--json", rrb_score],
            logs / "rrb-score.log")
        run([PY, "-m", "sentient.track", "bench", "--score", rrb_score, "--run-id", f"sentient-rrb-v{n}",
             "--name", f"Sentient Decision RoboRewardBench v{n}"], logs / "track-rrb.log")
    return questions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--test-source", action="append", default=[])
    ap.add_argument("--windows", default="6")
    ap.add_argument("--workers", default="12")
    ap.add_argument("--retention", default="0.2")
    ap.add_argument("--epochs", default="2")
    ap.add_argument("--rrb", help="RoboRewardBench test.jsonl (sentient.roborewardbench build); benched each version")
    ap.add_argument("--grow", type=float, default=1.5)
    ap.add_argument("--wait", type=float, default=1800, help="seconds between cycles that did not train")
    ap.add_argument("--start", type=int, default=0, help="first version number (default: after the last mix)")
    ap.add_argument("--last-trained", type=int, default=0, help="training questions of the last trained version")
    a = ap.parse_args()
    n = a.start or (max(version_numbers(Path(a.work) / "mix"), default=0) + 1)
    last = a.last_trained
    while True:
        trained = cycle(a, n, last)
        n += 1
        if trained == last:
            time.sleep(a.wait)
        last = trained


if __name__ == "__main__":
    main()
