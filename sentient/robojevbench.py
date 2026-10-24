"""RoboJev's benchmark as RoboJev reports it (josefchen/RoboJev-27B, benchmark/README.md and paper section 4): did the
robot complete its task, as P(success), scored by AUROC and balanced accuracy at 0.5, with ECE (10 bins) and Brier.

    python -m sentient.robojevbench build --raw ~/sd-data/raw --out ~/sd-data/robojev
    python -m sentient.bench predict --model <folder> --data ~/sd-data/robojev/test.jsonl --out preds/robojev-<name>.jsonl
    python -m sentient.robojevbench score preds/robojev-*.jsonl --data ~/sd-data/robojev/test.jsonl

Sets: "so101", 268 ArmnetBench policy rollouts (bench_v1.jsonl; split test_seen for held-out sessions of the six
trained tasks, test_unseen for the two held-out tasks), and "unseen", 459 RBM-1M OOD rollouts from eight lab/robot
domains (bench_ood_v1.jsonl). RoboJev's third set, 135 community clips, has no published frames or labels.

Protocol dense-v1, RoboJev's own: ten frames of the primary camera spread evenly from the first to the last, plus the
last frame of each other camera, at a 288 pixel short side, JPEG quality 92. The test files are read at the release
RoboJev's paper describes, so the episodes are exactly theirs.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

REVISION = "e089a9b69a365744b80765057d86583647af95a9"
FILES = {"so101": "bench_v1.jsonl", "unseen": "bench_ood_v1.jsonl"}
QUESTION = "Did the robot complete the task?"
FRAMES = 10
SHORT_SIDE = 288
# RoboJev-27B release summary (summary.json, 2026-10-23); APIs read four frames (clip-v1), RoboJev and JEV ten.
REFERENCE = {
    "RoboJev-27B": {"so101": (0.9932, 0.952), "unseen": (0.8904, 0.7816)},
    "JEV-27B-VL (ten frames)": {"unseen": (0.900, None)},
    "GPT-6.1 Sol": {"so101": (0.9891, 0.9284), "unseen": (0.8654, 0.7908)},
    "GPT-6 Astra": {"so101": (0.9869, 0.9329), "unseen": (0.8622, 0.7785)},
    "Claude Opus 5.5": {"so101": (0.9777, 0.9186), "unseen": (0.8615, 0.7393)},
    "TypeSafe Jev 1.14": {"so101": (0.9750, 0.8288), "unseen": (0.8408, 0.7208)},
}


def _fetch(name: str, dest: Path) -> Path:
    if not dest.exists():
        url = f"https://huggingface.co/josefchen/RoboJev-27B/resolve/{REVISION}/benchmark/{name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return dest


def _save(img, path: Path) -> None:
    img = img.convert("RGB")
    k = SHORT_SIDE / min(img.size)
    if k < 1:
        img = img.resize((round(img.width * k), round(img.height * k)))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=92)


def _duration(path: Path) -> float:
    import av
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration and s.time_base else float(c.duration or 0) / 1e6


def build(a):
    from .data.video import VideoFrames

    raw, out = Path(a.raw), Path(a.out)
    roots = {"so101": raw / "armnetbench", "unseen": raw / "robometer-ood"}
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "test.jsonl", "w") as f:
        for name, file in FILES.items():
            for line in open(_fetch(file, roots[name] / f"robojev_{file}")):
                row = json.loads(line)
                videos = {c: roots[name] / p for c, p in row["videos"].items()}
                primary = "front"
                d = _duration(videos[primary])
                ts = [min(d - 1e-3, d * i / (FRAMES - 1)) for i in range(FRAMES)]
                stem = row["key"].replace("/", "_")
                got = VideoFrames(videos[primary]).frames(ts)
                rels, lines = [], []
                for i, t in enumerate(ts):
                    rel = f"img/{name}/{stem}-{i:02d}.jpg"
                    if not (out / rel).exists():
                        _save(got[t], out / rel)
                    rels.append(rel)
                    lines.append(f"t={t:.1f}s: <image>")
                state = f"Task: {row['instruction'].strip()}\nFrames from the {primary} camera, first to last:\n" + \
                    "\n".join(lines)
                for cam in sorted(c for c in videos if c != primary):
                    dc = _duration(videos[cam])
                    rel = f"img/{name}/{stem}-{cam}-last.jpg"
                    if not (out / rel).exists():
                        _save(VideoFrames(videos[cam]).frames([dc - 1e-3])[dc - 1e-3], out / rel)
                    rels.append(rel)
                    state += f"\nLast frame from the {cam} camera: <image>"
                f.write(json.dumps({"state": state, "images": rels,
                                    "questions": {"done": {"type": "noul", "instructions": QUESTION}},
                                    "targets": {"done": {"true" if row["label"] == "successful" else "false": 1.0}},
                                    "meta": {"source": f"robojev_{name}", "set": name, "split": row["split"],
                                             "domain": row["task_id"], "episode": row["key"]}}) + "\n")
                n += 1
    print(n, "episodes written to", out / "test.jsonl")


def _p_success(answer: dict) -> float:
    if "noul" in answer:
        return float(answer["noul"])
    probs = answer.get("probabilities") or {}
    return float(probs.get("true", probs.get("yes", 0.0)))


def metrics(rows: list[tuple[float, int]]) -> dict:
    pos = [p for p, y in rows if y]
    neg = [p for p, y in rows if not y]
    if not pos or not neg:
        return {"n": len(rows)}
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    bacc = (sum(p >= 0.5 for p in pos) / len(pos) + sum(p < 0.5 for p in neg) / len(neg)) / 2
    ece = 0.0
    for b in range(10):
        sel = [(p, y) for p, y in rows if b / 10 <= p < (b + 1) / 10 or (b == 9 and p == 1.0)]
        if sel:
            ece += len(sel) / len(rows) * abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return {"n": len(rows), "auroc": round(wins / (len(pos) * len(neg)), 4), "balanced_accuracy": round(bacc, 4),
            "ece": round(ece, 4), "brier": round(sum((p - y) ** 2 for p, y in rows) / len(rows), 4)}


def score(a):
    data = [json.loads(l) for l in open(a.data)]
    report = {}
    for path in a.preds:
        groups = defaultdict(list)
        for line in open(path):
            pr = json.loads(line)
            if "error" in pr or "done" not in pr.get("answers", {}):
                continue
            meta = data[pr["i"]]["meta"]
            row = (_p_success(pr["answers"]["done"]), int("true" in data[pr["i"]]["targets"]["done"]))
            groups[meta["set"]].append(row)
            if meta["set"] == "so101":
                groups[f"so101_{meta['split']}"].append(row)
            else:
                groups[f"unseen_{meta['domain']}"].append(row)
        report[Path(path).stem] = {g: metrics(r) for g, r in sorted(groups.items())}
    ref = {m: {s: {"auroc": v[0], "balanced_accuracy": v[1]} for s, v in sets.items()} for m, sets in REFERENCE.items()}
    print(json.dumps({"reference": ref, **report}, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("preds", nargs="+")
    s.add_argument("--data", required=True)
    s.add_argument("--json")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
