"""BodenAI/RW-RL-HIL-Dataset (CC-BY-4.0): 3,347 real deployments of a policy on a Galaxea R1-Lite at home tasks,
where an operator took over whenever the policy could not cope and handed control back after recovering.

Labels used: every takeover as a (start, end) range in seconds (the saved annotations are complete: 279 episodes have
none). The moment an experienced operator takes over is a person's judgement that the robot was about to fail, so
these episodes teach the takeover question. No success or progress is claimed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..episode import Episode
from ..video import VideoFrames

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def items(raw: Path, shard: int, shards: int):
    root = Path(raw) / "bodenai-hil"
    for task_dir in sorted(p for p in root.iterdir() if (p / "meta" / "info.json").exists()):
        info = json.loads((task_dir / "meta" / "info.json").read_text())
        fps = float(info["fps"])
        pattern = info["video_path"]
        for line in open(task_dir / "meta" / "episodes.jsonl"):
            ep = json.loads(line)
            idx = int(ep["episode_index"])
            if int(hashlib.sha256(f"{task_dir.name}:{idx}".encode()).hexdigest()[:8], 16) % shards != shard:
                continue
            videos = {}
            for cam in CAMERAS:
                p = task_dir / pattern.format(episode_chunk=idx // int(info.get("chunks_size", 1000)),
                                              video_key=f"observation.images.{cam}", episode_index=idx)
                if p.exists():
                    videos[cam] = p
            if not videos:
                continue
            spans = [(r["start_frame"] / fps, (r["end_frame"] + 1) / fps) for r in ep.get("intervention_ranges") or []]
            yield Episode(source="bodenai", episode_id=f"{task_dir.name}:{idx}", family=f"bodenai:{task_dir.name}:{idx}",
                          task=ep["tasks"][0], duration=ep["length"] / fps, cameras=list(videos),
                          frame=lambda c, t, v=videos: VideoFrames(v[c]).frames([t])[t],
                          frames=lambda c, ts, v=videos: VideoFrames(v[c]).frames(ts),
                          robot="Galaxea R1-Lite (two arms, mobile base)", interventions=spans)
