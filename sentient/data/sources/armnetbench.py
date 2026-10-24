"""armnet/armnetbench_v01_robometer (Apache-2.0): 3,718 SO-101 episodes (single-arm and bimanual) over 12 tasks, from
7 policies plus human teleoperation, each labelled successful, suboptimal or failure, with three cameras.

RoboJev's SO-101 benchmark is drawn from it: its 40 test recording sessions and the two tasks it never trains on
(cable_unclip, ring_insert) are test only here, so a model trained on this source is still judged on unseen sessions
and unseen tasks exactly as RoboJev is. The test sessions are read from RoboJev's bench_v1.jsonl, kept beside the
data as robojev_bench_v1.jsonl (sentient.robojevbench build fetches it).

Questions: the outcome where it is unambiguous (a suboptimal rollout is not asked whether it succeeded) and the
quality as a three-level episode score, over frames spread across the whole episode.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..episode import Episode
from ..video import VideoFrames

LEVELS = ["failure", "suboptimal", "successful"]
HELD_OUT_TASKS = {"cable_unclip", "ring_insert"}


def test_sessions(root: Path) -> set[str]:
    bench = Path(root) / "robojev_bench_v1.jsonl"
    if not bench.exists():
        raise SystemExit(f"{bench} is missing: run sentient.robojevbench build first, so its test sessions stay out")
    return {json.loads(l)["session"] for l in open(bench)}


def items(raw: Path, shard: int, shards: int):
    import av
    import pyarrow.parquet as pq

    root = Path(raw) / "armnetbench"
    held = test_sessions(root)
    for robot, config in (("SO-101 arm", "so101"), ("two SO-101 arms", "bimanual_so101")):
        rows = pq.read_table(root / config / "train-00000-of-00001.parquet",
                             columns=["id", "task", "frames", "quality_label"]).to_pylist()
        episodes = defaultdict(dict)
        for r in rows:            # one row per camera: armnetbench/<task>/<policy>/<session>/<episode>/<camera>
            key, camera = r["id"].rsplit("/", 1)
            episodes[key][camera] = r
        for n, (key, cams) in enumerate(sorted(episodes.items())):
            if n % shards != shard:
                continue
            _, task_id, policy, session, _ = key.split("/")
            first = next(iter(cams.values()))
            quality = first["quality_label"]
            if quality not in LEVELS:
                continue
            videos = {c: root / r["frames"] for c, r in cams.items() if (root / r["frames"]).exists()}
            if not videos:
                continue
            main = "front" if "front" in videos else "top" if "top" in videos else next(iter(videos))
            duration = 0.0
            try:
                with av.open(str(videos[main])) as c:
                    s = c.streams.video[0]
                    duration = float(s.duration * s.time_base) if s.duration and s.time_base else float(c.duration or 0) / 1e6
            except Exception:
                continue
            if duration <= 0.5:
                continue
            yield Episode(source="armnetbench", episode_id=key, family=f"armnetbench:{session}",
                          task=first["task"].strip(), duration=duration,
                          cameras=[main] + sorted(c for c in videos if c != main),
                          frame=lambda c, t, v=videos: VideoFrames(v[c]).frames([t])[t],
                          frames=lambda c, ts, v=videos: VideoFrames(v[c]).frames(ts),
                          robot=robot, success=None if quality == "suboptimal" else quality == "successful",
                          success_from="human", ends_when_done=False,
                          episode_score=(LEVELS.index(quality), LEVELS), episode_score_from="human",
                          designated_test=session in held or task_id in HELD_OUT_TASKS,
                          extra={"forecastable": False, "whole_episode": True, "policy": policy})
