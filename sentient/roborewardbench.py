"""RoboRewardBench as FLORA reports it: all 2,831 human-scored test episodes of teetone/RoboReward, one 1-5 outcome
score per episode, mean absolute error on the 1-5 scale x 100 (FLORA-Rewarder 40.0, GPT-6 Astra 55.4, RoboReward-8B
66.5; FLORA paper, Table 4 and Appendix C). The headline is RoboReward's "overall": the mean of the per-dataset MAEs,
not the mean over episodes (both are reported).

    python -m sentient.roborewardbench build --raw ~/sd-data/raw --out ~/sd-data/rrb
    python -m sentient.bench predict --model <folder> --data ~/sd-data/rrb/test.jsonl --out preds/rrb-<name>.jsonl
    python -m sentient.roborewardbench score preds/rrb-*.jsonl --data ~/sd-data/rrb/test.jsonl

Frames: min(16, max(2, floor(10 T))) spread evenly over the whole video, first to last (FLORA reads up to 64; the
server takes 16 images per request). One fixed question with the RoboReward rubric. A model's probabilities over the
five levels are decoded to one score by their median, the decoding that minimises expected absolute error; the most
likely level is reported beside it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

RUBRIC = ["1: no success. The task was not accomplished and no meaningful progress toward it was made.",
          "2: minimal progress. Some relevant motion or contact, but far from completing the task.",
          "3: partial completion. Part of the task was done, but key parts remain.",
          "4: near completion. The task is almost done, with a small part missing or imperfect.",
          "5: perfect completion. The task was fully accomplished."]
QUESTION = ("Score how well the robot accomplished the task by the end of the episode, judging from the frames "
            "against the rubric.")


def build(a):
    from .data.build import _save, whole_episode
    from .data.video import VideoFrames

    import av
    base = Path(a.raw) / "roboreward" / "test"
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "test.jsonl", "w") as f:
        for line in open(base / "metadata.jsonl"):
            row = json.loads(line)
            path = base / row["file_name"]
            with av.open(str(path)) as c:
                s = c.streams.video[0]
                d = float(s.duration * s.time_base) if s.duration and s.time_base else float(c.duration or 0) / 1e6
            ts = whole_episode(d, 16)
            got = VideoFrames(path).frames(ts)
            rels = []
            for i, t in enumerate(ts):
                rel = f"img/{Path(row['file_name']).stem}-{i:02d}.jpg"
                if not (out / rel).exists():
                    _save(got[t], out / rel, a.side)
                rels.append(rel)
            state = f"Task: {row['task'].strip()}\nFrames from the rollout, first to last:\n" + "\n".join(
                f"t={t:.1f}s: <image>" for t in ts)
            f.write(json.dumps({"state": state, "images": rels,
                                "questions": {"score": {"type": "score", "instructions": QUESTION, "criteria": RUBRIC}},
                                "targets": {"score": {str(int(row["reward"]) - 1): 1.0}},
                                "meta": {"source": "roboreward_bench", "episode": row["file_name"],
                                         "dataset": row["file_name"].split("/")[0], "split": "test",
                                         "kinds": ["episode_score"], "frames": len(ts)}}) + "\n")
            n += 1
    print(n, "episodes written to", out / "test.jsonl")


def _levels(answer: dict) -> list[float]:
    probs = answer.get("probabilities") or {}
    p = [float(probs.get(str(i), 0.0)) for i in range(5)]
    total = sum(p) or 1.0
    return [v / total for v in p]


def score(a):
    data = [json.loads(l) for l in open(a.data)]
    report = {}
    for path in a.preds:
        err_med = err_top = 0.0
        n = 0
        per = defaultdict(lambda: [0.0, 0])
        for line in open(path):
            pr = json.loads(line)
            if "error" in pr or "score" not in pr.get("answers", {}):
                continue
            rec = data[pr["i"]]
            truth = int(next(iter(rec["targets"]["score"]))) + 1
            p = _levels(pr["answers"]["score"])
            cum, median = 0.0, 5
            for i, v in enumerate(p):
                cum += v
                if cum >= 0.5:
                    median = i + 1
                    break
            top = max(range(5), key=lambda i: p[i]) + 1
            err_med += abs(median - truth)
            err_top += abs(top - truth)
            per[rec["meta"]["dataset"]][0] += abs(median - truth)
            per[rec["meta"]["dataset"]][1] += 1
            n += 1
        groupwise = sum(e / c for e, c in per.values()) / max(len(per), 1)
        report[Path(path).stem] = {"roborewardbench": {"episodes": n, "of": len(data),
                                                       "mae_x100": round(100 * groupwise, 1),
                                                       "mae_x100_pooled_median": round(100 * err_med / max(n, 1), 1),
                                                       "mae_x100_pooled_top": round(100 * err_top / max(n, 1), 1)},
                                   "by_dataset": {k: round(100 * e / c, 1) for k, (e, c) in sorted(per.items())}}
    print(json.dumps({"reference": {"FLORA-Rewarder": 40.0, "GPT-6 Astra": 55.4, "RoboReward-8B": 66.5},
                      **report}, indent=1))
    if a.json:
        json.dump(report, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--side", type=int, default=448, help="long side of each frame in pixels")
    s = sub.add_parser("score")
    s.add_argument("preds", nargs="+")
    s.add_argument("--data", required=True)
    s.add_argument("--json")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
