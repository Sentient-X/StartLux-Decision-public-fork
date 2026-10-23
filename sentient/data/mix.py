"""Mix built sources into one training set: per-source caps, a retention share, a seeded shuffle, and a manifest.

    python -m sentient.data.mix --built ~/sd-data/built --out ~/sd-data/mix/v0 \\
        --source fineart:20000 --source failtime --source roboarena:15000 --retention 0.2

Image paths are rewritten relative to the output folder. dev.jsonl (for calibrate.py and the fine-tuner's dev loss)
and test.jsonl (the bench) are kept per source; nothing from a test split ever enters train.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def _rebase(rec: dict, src: Path, out: Path) -> dict:
    rel = os.path.relpath(src, out)
    if rec.get("images"):
        rec = dict(rec, images=[os.path.join(rel, p) for p in rec["images"]])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--built", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", action="append", required=True, help="name[:max train records]")
    ap.add_argument("--retention", type=float, default=0.2, help="share of retention questions in train")
    ap.add_argument("--dev-per-source", type=int, default=1500)
    ap.add_argument("--test-per-source", type=int, default=400, help="records per source in the bench set")
    ap.add_argument("--seed", default="v0")
    a = ap.parse_args()
    built, out = Path(a.built), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, dev, test, manifest = [], [], [], {"sources": {}}
    for spec in a.source:
        name, _, cap = spec.partition(":")
        src = built / name
        rows = {s: [_rebase(r, src, out) for r in _read(src / f"{s}.jsonl")] for s in ("train", "dev", "test")}
        random.Random(f"sentient.mix/{a.seed}:{name}").shuffle(rows["train"])
        for s in ("dev", "test"):         # ranked by a hash of the record, so the picks stay put as data grows
            rows[s].sort(key=lambda r: hashlib.sha256(
                json.dumps([r["meta"]["episode"], r["meta"].get("t"), sorted(r["questions"])]).encode()).hexdigest())
        if cap:
            rows["train"] = rows["train"][:int(cap)]
        rows["dev"] = rows["dev"][:a.dev_per_source]
        rows["test"] = rows["test"][:a.test_per_source]
        assert not {r["meta"]["family"] for r in rows["train"]} & {r["meta"]["family"] for r in rows["test"]}, name
        train += rows["train"]
        dev += rows["dev"]
        test += rows["test"]
        manifest["sources"][name] = {s: len(v) for s, v in rows.items()}
    ret = built / "retention"
    if a.retention > 0:                   # a share of training questions; retention records hold one question each
        robot_questions = sum(len(r["questions"]) for r in train)
        n = int(robot_questions * a.retention / (1 - a.retention))
        pool = _read(ret / "train.jsonl")
        random.Random(f"sentient.mix/{a.seed}:retention").shuffle(pool)
        train += pool[:n]
        dev += _read(ret / "dev.jsonl")[:a.dev_per_source]
        manifest["sources"]["retention"] = {"train": min(n, len(pool))}
    random.Random(f"sentient.mix/{a.seed}").shuffle(train)
    for split, rows in (("train", train), ("dev", dev), ("test", test)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[split] = {"records": len(rows), "questions": sum(len(r["questions"]) for r in rows),
                           "sha256": hashlib.sha256((out / f"{split}.jsonl").read_bytes()).hexdigest()}
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
