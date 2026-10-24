"""Robometer's out-of-distribution ranking benchmark (RBM-EVAL-OOD, robometer/rbm-1m-ood): within each task, rollouts
labelled failure < suboptimal < successful should be ranked by the model's reward in that order. Robometer reports
Kendall tau-a per task, averaged over the tasks of a dataset, then equally over datasets (Robometer 64-66,
RynnValue-8B 70.4, FLORA-Rewarder 72.0, GPT-6 Astra 82.3; as tabled in FLORA, Table 5).

    python -m sentient.robometerbench build --raw ~/sd-data/raw --out ~/sd-data/robometer
    python -m sentient.bench predict --model <folder> --data ~/sd-data/robometer/test.jsonl --out preds/robometerbench-<name>.jsonl
    python -m sentient.robometerbench score preds/robometerbench-*.jsonl --data ~/sd-data/robometer/test.jsonl

The reward is the expected 1-5 score under RoboRewardBench's rubric, read from frames spread over the whole rollout
(as sentient.roborewardbench asks it), so one question serves both benchmarks. The published sizes disagree (paper
976 rollouts, the Hugging Face release 571, FLORA 567); this uses every labelled rollout of the release. Two numbers:
tau-a over all pairs in a task (ties in the label count in the denominator, Kendall's definition) and tau over pairs
with different labels only; which one the papers table is not stated, so both are kept.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .roborewardbench import QUESTION, RUBRIC

REFERENCE = {"GPT-6 Astra": 82.3, "FLORA-Rewarder": 72.0, "RynnValue-8B-Quantile": 70.4, "Robometer-4B": 65.5}


def build(a):
    from .data.build import _save, whole_episode
    from .data.sources import robometer_ood

    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "test.jsonl", "w") as f:
        for ep in robometer_ood.items(Path(a.raw), 0, 1):
            ts = whole_episode(ep.duration, 16)
            got = ep.frames("camera", ts)
            rels = []
            for i, t in enumerate(ts):
                rel = f"img/{ep.episode_id}-{i:02d}.jpg"
                if not (out / rel).exists():
                    _save(got[t], out / rel, a.side)
                rels.append(rel)
            state = f"Task: {ep.task.strip()}\nFrames from the rollout, first to last:\n" + "\n".join(
                f"t={t:.1f}s: <image>" for t in ts)
            level = ep.episode_score[0] if ep.episode_score else 0
            f.write(json.dumps({"state": state, "images": rels,
                                "questions": {"score": {"type": "score", "instructions": QUESTION, "criteria": RUBRIC}},
                                "targets": {"score": {"0": 1.0}},      # unused: the label is the quality level below
                                "meta": {"source": "robometer_bench", "episode": ep.episode_id, "dataset": ep.robot,
                                         "task": ep.task.strip(), "quality": level, "frames": len(ts)}}) + "\n")
            n += 1
    print(n, "rollouts written to", out / "test.jsonl")


def _reward(answer: dict) -> float:
    probs = answer.get("probabilities") or {}
    p = [float(probs.get(str(i), 0.0)) for i in range(5)]
    total = sum(p) or 1.0
    return sum((i + 1) * v for i, v in enumerate(p)) / total


def taus(rows: list[tuple[float, int]]) -> tuple[float, float] | None:
    """-> (tau-a over all pairs, tau over pairs with different labels) for one task's (reward, level) rows."""
    n = len(rows)
    if n < 2:
        return None
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            dl = rows[i][1] - rows[j][1]
            dr = rows[i][0] - rows[j][0]
            if dl and dr:
                c += (dl > 0) == (dr > 0)
                d += (dl > 0) != (dr > 0)
    differing = sum(1 for i in range(n) for j in range(i + 1, n) if rows[i][1] != rows[j][1])
    if not differing:
        return None
    return (c - d) / (n * (n - 1) / 2), (c - d) / differing


def score(a):
    data = [json.loads(l) for l in open(a.data)]
    report = {}
    for path in a.preds:
        tasks = defaultdict(list)
        for line in open(path):
            pr = json.loads(line)
            if "error" in pr or "score" not in pr.get("answers", {}):
                continue
            m = data[pr["i"]]["meta"]
            tasks[(m["dataset"], m["task"])].append((_reward(pr["answers"]["score"]), m["quality"]))
        per_dataset = defaultdict(list)
        for (dataset, _), rows in tasks.items():
            t = taus(rows)
            if t:
                per_dataset[dataset].append(t)
        by = {k: (sum(t[0] for t in v) / len(v), sum(t[1] for t in v) / len(v)) for k, v in sorted(per_dataset.items())}
        report[Path(path).stem] = {
            "robometer_ood": {"rollouts": sum(len(r) for r in tasks.values()), "datasets": len(by),
                              "kendall_tau_a_x100": round(100 * sum(v[0] for v in by.values()) / max(len(by), 1), 1),
                              "kendall_differing_x100": round(100 * sum(v[1] for v in by.values()) / max(len(by), 1), 1)},
            "by_dataset_tau_a": {k: round(100 * v[0], 1) for k, v in by.items()},
            "by_dataset_differing": {k: round(100 * v[1], 1) for k, v in by.items()}}
    print(json.dumps({"reference_kendall_x100": REFERENCE, **report}, indent=1))
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
