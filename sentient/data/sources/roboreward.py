"""teetone/RoboReward (CC-BY-4.0): real-robot rollout videos (Open X-Embodiment and RoboArena) with an end-of-episode
progress score 1-5 under a stated rubric. Train and val hold generated negatives (another task swapped in for the
same video, or a successful video cut short); test is RoboRewardBench, fully human-verified, designated test.

Variants of one rollout (counterfactual tasks "_attempt_N_score_N", clips "_N") share a family, and every variant of
a rollout in RoboRewardBench is test only, so nothing about a bench episode is ever trained on.
Questions: the 1-5 score at the end, and success where the score is unambiguous (5 done; 1-3 not done; 4 is not asked).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..episode import Episode
from ..video import VideoFrames

RUBRIC = ["no success", "minimal progress", "partial completion", "near completion", "perfect completion"]
_ROLLOUT = re.compile(r"^(.*_index_\d+)")             # the source episode; anything after it names a variant


def _family(file_name: str) -> str:
    stem = Path(file_name).stem
    if file_name.startswith("robo_arena/"):
        return "roboarena:" + stem.split("_")[0]          # the RoboArena session, shared with sentient's roboarena source
    m = _ROLLOUT.match(stem)
    return "roboreward:" + (m.group(1) if m else stem)


def bench_sessions(raw: Path) -> set[str]:
    """RoboArena sessions in RoboRewardBench: the roboarena source keeps them out of training."""
    meta = Path(raw) / "roboreward" / "test" / "metadata.jsonl"
    if not meta.exists():
        return set()
    return {Path(json.loads(l)["file_name"]).stem.split("_")[0] for l in open(meta)
            if json.loads(l)["file_name"].startswith("robo_arena/")}


def items(raw: Path, shard: int, shards: int):
    import av

    meta = Path(raw) / "roboreward" / "test" / "metadata.jsonl"
    bench = {_family(json.loads(l)["file_name"]) for l in open(meta)} if meta.exists() else set()
    for split in ("train", "val", "test"):
        base = Path(raw) / "roboreward" / split
        meta = base / "metadata.jsonl"
        if not meta.exists():
            continue
        for n, line in enumerate(open(meta)):
            if n % shards != shard:
                continue
            row = json.loads(line)
            path = base / row["file_name"]
            if not path.exists():
                continue
            duration = 0.0
            try:
                with av.open(str(path)) as c:
                    s = c.streams.video[0]
                    duration = float(s.duration * s.time_base) if s.duration and s.time_base else float(c.duration or 0) / 1e6
            except Exception:
                continue
            if duration <= 0.3:
                continue
            score = int(row["reward"]) - 1
            yield Episode(source="roboreward", episode_id=f"{split}:{Path(row['file_name']).stem}",
                          family=_family(row["file_name"]), task=row["task"].strip(), duration=duration,
                          cameras=["camera"], frame=lambda c, t, p=path: VideoFrames(p).frames([t])[t],
                          frames=lambda c, ts, p=path: VideoFrames(p).frames(ts),
                          success=True if score == 4 else False if score <= 2 else None,
                          success_from="human" if split == "test" else "program", ends_when_done=False,
                          episode_score=(score, RUBRIC),
                          episode_score_from="human" if split == "test" else "program",
                          designated_test=split == "test" or _family(row["file_name"]) in bench,
                          extra={"forecastable": False, "whole_episode": True})   # clips may be cut short of what really happened next
