"""RoboArena dumps (MIT): real DROID-platform policy rollouts, scored by the person who ran them.

Per rollout: binary success and a 0-1 partial-success score (human), the instruction, and up to three videos
(left/right exterior, wrist). Per session: the evaluator's preference between policies A and B ("A", "B", "TIE")
with written feedback. Sessions with more than two policies give soft pairwise preferences from their partial
scores. Both dumps are read; a session appearing in both is read once. Episodes and preferences of one session share
a family, so a session never straddles splits.
"""
from __future__ import annotations

import itertools
from pathlib import Path

from ..episode import Episode, Preference
from ..targets import yes_probability
from ..video import VideoFrames

DUMPS = ["roboarena-0717", "roboarena-0203"]
CAMS = {"video_left": "left exterior camera", "video_right": "right exterior camera", "video_wrist": "wrist camera"}


def _rollout(session: str, label: str, folder: Path, meta: dict, task: str) -> Episode | None:
    videos = {}
    for f in folder.glob("*.mp4"):
        for key, name in CAMS.items():
            if f.name.endswith(key + ".mp4"):
                videos[name] = f
    if not videos:
        return None
    import av
    duration = 0.0
    try:
        with av.open(str(next(iter(videos.values())))) as c:
            s = c.streams.video[0]
            duration = float(s.duration * s.time_base) if s.duration and s.time_base else float(c.duration or 0) / 1e6
    except Exception:
        return None
    if not duration or duration <= 1:
        return None
    partial = meta.get("partial_success")
    success = meta.get("binary_success")
    return Episode(source="roboarena", episode_id=f"{session}:{label}", family=f"roboarena:{session}", task=task,
                   duration=duration, cameras=list(videos),
                   frame=lambda c, t, v=videos: VideoFrames(v[c]).frames([t])[t],
                   frames=lambda c, ts, v=videos: VideoFrames(v[c]).frames(ts),
                   robot="Franka arm (DROID setup)",
                   success=None if success is None else bool(success), success_from="human", ends_when_done=False,
                   extra={"final_progress": None if partial is None else float(partial), "final_progress_sigma": 0.08})


def items(raw: Path, shard: int, shards: int):
    import hashlib

    import yaml

    from .roboreward import bench_sessions

    bench = bench_sessions(Path(raw))     # in RoboRewardBench: test only here too, so no claim on it leaks
    seen = set()
    for dump in DUMPS:
        base = Path(raw) / dump / "evaluation_sessions"
        if not base.exists():
            continue
        for folder in sorted(base.iterdir()):
            session = folder.name
            if session in seen or int(hashlib.sha256(session.encode()).hexdigest()[:8], 16) % shards != shard:
                continue
            seen.add(session)
            try:
                meta = yaml.safe_load(open(folder / "metadata.yaml"))
            except Exception:
                continue
            task = (meta.get("language_instruction") or "").strip()
            policies = meta.get("policies") or {}
            if not task or not policies:
                continue
            eps = {}
            for label, pm in policies.items():
                sub = next((d for d in folder.iterdir() if d.is_dir() and d.name.startswith(f"{label}_")), None)
                ep = _rollout(session, label, sub, pm, task) if sub else None
                if ep is not None:
                    ep.designated_test = session in bench
                    eps[label] = ep
                    yield ep
            pref = str(meta.get("preference") or "").upper()
            if len(policies) == 2 and set(eps) == {"A", "B"} and pref in ("A", "B", "TIE"):
                votes = {"A": (1.0, 0.0, 0.0), "B": (0.0, 1.0, 0.0), "TIE": (0.0, 0.0, 1.0)}[pref]
                yield Preference(source="roboarena", pair_id=f"{session}:AB", family=f"roboarena:{session}", task=task,
                                 a=eps["A"], b=eps["B"], votes=votes, designated_test=session in bench)
            elif len(eps) > 2:
                for x, y in itertools.combinations(sorted(eps), 2):
                    px, py = policies[x].get("partial_success"), policies[y].get("partial_success")
                    if px is None or py is None:
                        continue
                    d = float(px) - float(py)
                    tie = max(0.0, 1 - abs(d) / 0.1) if abs(d) < 0.1 else 0.0
                    win = yes_probability(d, 0.05)
                    yield Preference(source="roboarena", pair_id=f"{session}:{x}{y}", family=f"roboarena:{session}",
                                     task=task, a=eps[x], b=eps[y],
                                     votes=(win * (1 - tie), (1 - win) * (1 - tie), tie),
                                     designated_test=session in bench)
