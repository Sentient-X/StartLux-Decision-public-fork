"""lerobot/FineART (CC-BY-4.0): 40,543 teleoperated bimanual episodes with contiguous, human-written subtask segments.

Labels used: stages (human; each row active from its timestamp to the next), and SARM-style progress derived from
them: (completed stages + fraction of the current one) / stages, marked "program" because the even weighting of
stages is ours. Episodes are teleoperated demonstrations, so no failures or setbacks are claimed. Six whole tasks,
picked by hash, are held out as a novel-task test.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..episode import Episode, Stage
from ..video import opened

CAMERAS = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
HELD_OUT_TASKS = 6


def _held_out(tasks: list[str]) -> set[str]:
    ranked = sorted(tasks, key=lambda t: hashlib.sha256(f"sentient.fineart.task/v1:{t}".encode()).hexdigest())
    return set(ranked[:HELD_OUT_TASKS])


def _stage_rows(raw: Path) -> dict[int, list[tuple[float, str]]]:
    """episode -> [(start s, text)] from the first frame of each episode (the rows repeat on every frame), cached in
    <raw>/stages.json once every data file is present."""
    import json

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    cache = raw / "stages.json"
    files = sorted((raw / "data").glob("*/*.parquet"))
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["files"] == len(files):
            return {int(k): [tuple(r) for r in v] for k, v in saved["rows"].items()}
    out: dict[int, list[tuple[float, str]]] = {}
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "language_persistent"])
        t = t.filter(pc.equal(t["frame_index"], 0))
        for ep, rows in zip(t["episode_index"].to_pylist(), t["language_persistent"].to_pylist()):
            subs = sorted((float(r["timestamp"]), r["content"].strip()) for r in rows or []
                          if r.get("style") == "subtask" and r.get("content"))
            if subs:
                out[int(ep)] = subs
    cache.write_text(json.dumps({"files": len(files), "rows": {str(k): v for k, v in out.items()}}))
    return out


def items(raw: Path, shard: int, shards: int):
    import pyarrow.parquet as pq

    raw = Path(raw) / "fineart"
    cols = ["episode_index", "tasks", "length"] + [f"videos/observation.images.{c}/{k}" for c in CAMERAS
                                                   for k in ("chunk_index", "file_index", "from_timestamp")]
    meta = pq.read_table(raw / "meta/episodes/chunk-000/file-000.parquet", columns=cols).to_pylist()
    rows = _stage_rows(raw)
    held = _held_out(sorted({m["tasks"][0] for m in meta}))
    for m in meta:
        ep = int(m["episode_index"])
        if ep % shards != shard or ep not in rows:
            continue
        cams = []
        for c in CAMERAS:
            ci, fi = m.get(f"videos/observation.images.{c}/chunk_index"), m.get(f"videos/observation.images.{c}/file_index")
            if ci is None:
                continue
            path = raw / f"videos/observation.images.{c}/chunk-{ci:03d}/file-{fi:03d}.mp4"
            if path.exists():
                cams.append((c, str(path), float(m[f"videos/observation.images.{c}/from_timestamp"])))
        if not cams:
            continue
        duration = m["length"] / 30.0
        subs = rows[ep]
        stages = [Stage(s, (subs[i + 1][0] if i + 1 < len(subs) else duration), text) for i, (s, text) in enumerate(subs)]
        stages = [s for s in stages if s.end > s.start]
        if len(stages) < 2:
            continue
        paths = {c: (p, off) for c, p, off in cams}

        def progress(t: float, stages=stages) -> float:
            if t < stages[0].start:
                return 0.0
            for i, s in enumerate(stages):
                if t < s.end:
                    return (i + (t - s.start) / (s.end - s.start)) / len(stages)
            return 1.0

        def frame(camera: str, t: float, paths=paths):
            p, off = paths[camera]
            return opened(p, off).frame(t)

        task = m["tasks"][0]
        yield Episode(source="fineart", episode_id=str(ep), family=f"fineart:{ep}", task=task, duration=duration,
                      cameras=list(paths), frame=frame, robot="bimanual ALOHA-style arms (Trossen)",
                      stages=stages, stages_from="human", progress=progress, progress_from="program",
                      progress_sigma=0.08, designated_test=task in held)
