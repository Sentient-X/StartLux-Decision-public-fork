"""Sentient Reward Bench: ask a decision model the held-out robot questions and score its probabilities.

    python -m sentient.bench predict --model /path/to/folder --data test.jsonl --out preds/base.jsonl
    python -m sentient.bench predict --openrouter typesafe/jev-1.14 --data test.jsonl --out preds/jev.jsonl
    python -m sentient.bench score preds/*.jsonl

Records are in the fine-tuning format (v1/docs/finetuning.md) with images. In-process models answer through the v1
engine's decide(), as calibrate.py reads them. --openrouter sends OpenRouter's /api/v1/systemone the state as text and
image_url parts (the only form it delivers images in; other forms are dropped silently, so every request checks that
images were billed). Answers are read into labels with the upstream scorer's to_labels (v1/eval/suites.py).

Scores per question kind and per source: accuracy (top option equals the target's top option), log loss and Brier
against the soft target, and top-label ECE over 10 bins; "prior" is the score of always answering the training
class frequencies, which a useful model must beat.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "v1"))
sys.path.insert(0, str(ROOT / "v1" / "eval"))


def labels_of(q: dict, target: dict) -> dict:
    """A target in the fine-tuning format -> {label: p} in the upstream scorer's label set."""
    if q["type"] == "noul":
        return {"no": target.get("false", 0.0), "yes": target.get("true", 0.0)}
    return {str(k): float(v) for k, v in target.items()}


def _image_parts(state, images, base: Path) -> list:
    """State text and images as OpenRouter's state parts, each image at its <image> mark (else first)."""
    from startlux_decide.prompt import state_text
    text = state_text(state)
    uris = []
    for v in images:
        p = base / v
        uris.append("data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.is_file() else v)
    pieces = text.split("<image>")
    if len(pieces) - 1 != len(uris):
        pieces = [""] * (len(uris)) + [text]
    parts = []
    for i, piece in enumerate(pieces):
        if piece:
            parts.append({"type": "text", "text": piece})
        if i < len(uris):
            parts.append({"type": "image_url", "image_url": {"url": uris[i]}})
    return parts


def engine(a):
    if a.openrouter:
        import urllib.request
        key = os.environ["OPENROUTER_API_KEY"]

        def ask(state, questions, images, base):
            body = json.dumps({"model": a.openrouter, "state": _image_parts(state, images, base),
                               "questions": questions}).encode()
            req = urllib.request.Request("https://openrouter.ai/api/v1/systemone", body,
                                         {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                doc = json.loads(r.read())
            usage = doc.get("usage") or {}
            return doc["answers"], {"cost": usage.get("cost"),
                                    "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens")}
        return ask
    from startlux_decision import StartLuxDecision
    model = StartLuxDecision(a.model, max_pixels=a.max_pixels)

    def ask(state, questions, images, base):
        paths = [str(base / v) if (base / v).is_file() else v for v in images]
        return model.decide(state, questions, paths or None)[0], {}
    return ask


def predict(a):
    ask = engine(a)
    base = Path(a.data).resolve().parent
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["i"] for l in open(a.out)}
    with open(a.data) as f, open(a.out, "a") as out:
        for i, line in enumerate(f):
            if i in done or (a.limit and i >= a.limit):
                continue
            rec = json.loads(line)
            qs = {k: q for k, q in rec["questions"].items() if k in rec["targets"]}
            t = time.time()
            try:
                answers, usage = ask(rec["state"], qs, rec.get("images") or [], base)
            except Exception as e:
                out.write(json.dumps({"i": i, "error": repr(e)[:300]}) + "\n")
                continue
            out.write(json.dumps({"i": i, "answers": answers, "usage": usage, "seconds": round(time.time() - t, 3),
                                  "model": a.openrouter or a.model}) + "\n")
            out.flush()


def _metrics(rows):
    """rows: [(predicted {label: p}, target {label: p})] -> accuracy, log loss, Brier, ECE."""
    n = len(rows)
    acc = nll = brier = 0.0
    conf_ok = []
    for pred, target in rows:
        labels = sorted(set(pred) | set(target))
        total = sum(max(pred.get(l, 0.0), 0.0) for l in labels) or 1.0
        p = {l: max(pred.get(l, 0.0), 0.0) / total for l in labels}
        top_p = max(p, key=lambda l: p[l])
        top_t = max(target, key=lambda l: target[l])
        acc += top_p == top_t
        nll -= sum(target.get(l, 0.0) * math.log(max(p[l], 1e-6)) for l in labels)
        brier += sum((p[l] - target.get(l, 0.0)) ** 2 for l in labels)
        conf_ok.append((p[top_p], float(top_p == top_t)))
    ece = 0.0
    for b in range(10):
        sel = [(c, o) for c, o in conf_ok if b / 10 < c <= (b + 1) / 10 or (b == 0 and c == 0)]
        if sel:
            ece += len(sel) / n * abs(sum(c for c, _ in sel) / len(sel) - sum(o for _, o in sel) / len(sel))
    return {"n": n, "accuracy": round(acc / n, 4), "log_loss": round(nll / n, 4), "brier": round(brier / n, 4),
            "ece": round(ece, 4)}


def score(a):
    from suites import to_labels
    data = [json.loads(l) for l in open(a.data)]
    prior: dict = defaultdict(lambda: defaultdict(float))
    if a.train:                                   # class frequencies per kind, from the training split
        for line in open(a.train):
            rec = json.loads(line)
            for (k, q), kind in zip(rec["questions"].items(), rec["meta"]["kinds"]):
                for lab, p in labels_of(q, rec["targets"][k]).items():
                    prior[kind][lab] += p
    report = {}
    for path in a.preds:
        groups = defaultdict(list)
        priors = defaultdict(list)
        cost = errors = 0
        for line in open(path):
            pr = json.loads(line)
            if "error" in pr:
                errors += 1
                continue
            rec = data[pr["i"]]
            cost += (pr.get("usage") or {}).get("cost") or 0
            kinds = rec["meta"]["kinds"]
            for (k, q), kind in zip(rec["questions"].items(), kinds):
                if k not in pr["answers"]:
                    continue
                target = labels_of(q, rec["targets"][k])
                from startlux_decide.prompt import score_keys
                pred = to_labels(q, pr["answers"][k], score_keys)
                if q["type"] == "noul":
                    target = {"no": target["no"], "yes": target["yes"]}
                for g in ("all", f"kind:{kind}", f"source:{rec['meta']['source']}"):
                    groups[g].append((pred, target))
                    if prior.get(kind):
                        tot = sum(prior[kind].values())
                        priors[g].append(({l: prior[kind].get(l, 0.0) / tot for l in pred}, target))
        report[path] = {"errors": errors, "cost_usd": round(cost, 4),
                        **{g: {**_metrics(rows), **({"prior_log_loss": _metrics(priors[g])["log_loss"]} if priors.get(g) else {})}
                           for g, rows in sorted(groups.items())}}
    print(json.dumps(report, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model")
    p.add_argument("--openrouter", help="OpenRouter model id, e.g. typesafe/jev-1.14 (OPENROUTER_API_KEY)")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-pixels", type=int, default=1 << 20)
    s = sub.add_parser("score")
    s.add_argument("preds", nargs="+")
    s.add_argument("--data", required=True)
    s.add_argument("--train", help="training split, for the class-frequency prior")
    s.add_argument("--json")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    if a.cmd == "predict":
        if not (a.model or a.openrouter):
            ap.error("--model or --openrouter")
        predict(a)
    else:
        score(a)


if __name__ == "__main__":
    main()
