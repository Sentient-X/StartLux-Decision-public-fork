"""Turn episodes into fine-tuning records (v1/docs/finetuning.md): sample windows, write their frames, ask what the
labels can answer.

    python -m sentient.data.build --source fineart --raw ~/sd-data/raw --out ~/sd-data/built/fineart --windows 6

Writes <out>/{train,dev,test}.jsonl with image paths relative to <out>, and <out>/build.json with counts per split and
question kind. A record also carries "meta" (source, episode, family, frame times, kinds), which training ignores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import formats as F
from .episode import Episode, Preference
from .questions import GENERATORS

# How often each kind is asked when it applies: rarer, more informative kinds are favoured.
WEIGHT = {"progress": 1.0, "remaining": 0.3, "time_to_go": 0.4, "will_succeed": 0.8, "succeeded": 1.0, "stage": 1.0,
          "stage_progress": 0.6, "change": 1.2, "compare": 0.5, "failed": 1.2, "failure_type": 1.0, "takeover": 1.2,
          "episode_score": 1.0}
SIDES = [320, 384, 448, 512, 640]


def _seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _save(img, path: Path, side: int) -> None:
    img = img.convert("RGB")
    scale = side / max(img.size)
    if scale < 1:
        img = img.resize((max(32, round(img.width * scale)), max(32, round(img.height * scale))))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=88)


def window(ep: Episode, rng: random.Random) -> list[float]:
    """Frame times of one window, chronological: 1 to 8 frames ending at an anchor, sometimes the episode's end."""
    d = ep.duration
    end = d if rng.random() < 0.25 else rng.uniform(min(1.0, d), d)
    k = rng.choice([1, 2, 2, 3, 4, 4, 5, 6, 8])
    span = min(end, rng.choice([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, d]))
    if k == 1 or span <= 0:
        return [max(0.0, end - 1e-3)]
    return [max(0.0, end - span + span * i / (k - 1)) - (1e-3 if i == k - 1 else 0) for i in range(k)]


def episode_records(ep: Episode, out: Path, windows: int) -> list[dict]:
    """Plan every window first, then read each camera's frames in one pass, then write records."""
    plans = []
    for w in range(windows):
        rng = random.Random(_seed("sentient.build/v1", ep.source, ep.episode_id, w))
        ts = window(ep, rng)
        asked = [(k, GENERATORS[k](ep, ts, rng)) for k in GENERATORS]
        asked = [(k, q) for k, q in asked if q is not None]
        if not asked:
            continue
        rng.shuffle(asked)
        n = min(len(asked), rng.choice([1, 1, 2, 2, 3, 4]))
        picked = sorted(asked, key=lambda kq: -WEIGHT[kq[0]] * rng.random())[:n]
        cams = ep.cameras if len(ep.cameras) == 1 else rng.sample(ep.cameras, 2 if (rng.random() < 0.3 and len(ts) <= 4) else 1)
        side = rng.choice(SIDES[:3] if len(ts) * len(cams) > 4 else SIDES)
        plans.append((w, rng, ts, picked, cams, side))
    need: dict[str, set] = {}
    for _, _, ts, _, cams, side in plans:
        for c in cams:
            for t in ts:
                if not (out / _rel(ep, c, t, side)).exists():
                    need.setdefault(c, set()).add(t)
    images: dict[tuple[str, float], object] = {}
    try:
        for c, times in need.items():
            got = ep.frames(c, sorted(times)) if ep.frames else {t: ep.frame(c, t) for t in times}
            images.update({(c, t): img for t, img in got.items()})
    except Exception as e:  # an unreadable video drops the episode, not the build
        print(json.dumps({"skip": ep.episode_id, "error": repr(e)[:200]}), flush=True)
        return []
    recs = []
    for w, rng, ts, picked, cams, side in plans:
        frames = []
        for t in ts:
            for c in cams:
                rel = _rel(ep, c, t, side)
                if not (out / rel).exists():
                    if (c, t) not in images:
                        break
                    _save(images[(c, t)], out / rel, side)
                frames.append((c, t, rel))
        if len(frames) != len(ts) * len(cams):
            continue
        state, imgs = F.state_and_images(ep.task, frames, ep.robot, rng)
        questions, targets, used = {}, {}, set()
        for kind, (_, spec, target) in picked:
            key = F.key_name(kind, rng)
            while key in used:
                key += "_"
            used.add(key)
            questions[key], targets[key] = spec, target
        recs.append({"state": state, "images": imgs, "questions": questions, "targets": targets,
                     "meta": {"source": ep.source, "episode": ep.episode_id, "family": ep.family,
                              "split": ep.split(), "t": [round(t, 3) for t in ts], "kinds": [k for k, _ in picked]}})
    return recs


def _rel(ep: Episode, camera: str, t: float, side: int) -> str:
    return f"img/{ep.source}/{ep.episode_id}/{camera}-{t:.3f}-{side}.jpg".replace(" ", "_")


def preference_records(pref: Preference, out: Path) -> list[dict]:
    rng = random.Random(_seed("sentient.pref/v1", pref.source, pref.pair_id))
    k = rng.choice([2, 3, 4])
    side = rng.choice(SIDES[:3])
    frames = {}
    for name, ep in (("A", pref.a), ("B", pref.b)):
        cam = rng.choice(ep.cameras)
        ts = [ep.duration * (i + 1) / k - 1e-3 for i in range(k)]
        rels = []
        for t in ts:
            rel = f"img/{pref.source}/{ep.episode_id}/{cam}-{t:.3f}-{side}.jpg".replace(" ", "_")
            if not (out / rel).exists():
                _save(ep.frame(cam, t), out / rel, side)
            rels.append(rel)
        frames[name] = rels
    names = rng.choice([("A", "B"), ("first", "second"), ("rollout_1", "rollout_2"), ("left", "right")])
    task_line = rng.choice(F.TASK_LINES).format(task=pref.task, task_lc=pref.task)
    state = (task_line + f"\nRollout {names[0]}: " + " ".join("<image>" for _ in frames["A"])
             + f"\nRollout {names[1]}: " + " ".join("<image>" for _ in frames["B"]))
    va, vb, vt = pref.votes
    total = va + vb + vt
    with_tie = vt > 0 or rng.random() < 0.3
    target = {names[0]: va / total, names[1]: vb / total}
    if with_tie:
        target["tie"] = vt / total
    else:
        s = target[names[0]] + target[names[1]]
        target = {k: v / s for k, v in target.items()} if s > 0 else {names[0]: 0.5, names[1]: 0.5}
    text = F.phrase(["Which rollout did the task better?", "Two robots attempted the same task. Which one made more progress?",
                     "Compare the two rollouts: which is better?", "Which attempt would a careful reviewer prefer?"], rng)
    key = F.key_name("preference", rng)
    return [{"state": state, "images": frames["A"] + frames["B"],
             "questions": {key: {"type": "choice", "instructions": text,
                                 "criteria": F.ordered({k: None for k in target}, rng)}},
             "targets": {key: {k: round(v, 4) for k, v in target.items() if v > 1e-4}},
             "meta": {"source": pref.source, "episode": pref.pair_id, "family": pref.family, "split": pref.split(),
                      "kinds": ["preference"]}}]


def _work(args):
    adapter, raw, out, windows, shard, shards = args
    mod = importlib.import_module(f"sentient.data.sources.{adapter}")
    counts, files = Counter(), {}
    for item in mod.items(Path(raw), shard, shards):
        recs = preference_records(item, Path(out)) if isinstance(item, Preference) else episode_records(item, Path(out), windows)
        for r in recs:
            split = r["meta"]["split"]
            if split not in files:
                files[split] = open(Path(out) / f"{split}.{shard:03d}.jsonl", "a", encoding="utf-8")
            files[split].write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[split] += 1
            for k in r["meta"]["kinds"]:
                counts[f"{split}:{k}"] += 1
    for f in files.values():
        f.close()
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="adapter module in sentient.data.sources")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=6, help="windows per episode")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jsonl"):
        f.unlink()
    shards = a.workers * 4
    total = Counter()
    with ProcessPoolExecutor(a.workers) as pool:
        for c in pool.map(_work, [(a.source, a.raw, a.out, a.windows, s, shards) for s in range(shards)]):
            total.update(c)
    for split in ("train", "dev", "test"):
        parts = sorted(out.glob(f"{split}.*.jsonl"))
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as dst:
            for p in parts:
                dst.write(p.read_text(encoding="utf-8"))
                p.unlink()
    json.dump({"source": a.source, "windows": a.windows, "counts": dict(sorted(total.items()))},
              open(out / "build.json", "w"), indent=1)
    print(json.dumps(dict(sorted(total.items())), indent=1))


if __name__ == "__main__":
    main()
