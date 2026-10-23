"""FailTime-Short (CC-BY-4.0) labels over ViFailback recordings (MIT): AgileX Piper rollouts, each a success or a
failure with the second it went wrong, its type, and a written expected/observed/result reason (human).

Frames come from ViFailback's raw HDF5 (JPEG per frame, three cameras). The recordings carry no timestamps; frame
k is read as k / 25 s, the rate of ViFailback's own video export (and of its one-second annotation stride).
FailTime-Short's test split is designated test; FailTime-Long is never read here (test only).
"""
from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..episode import Episode

FPS = 25.0
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def _observed(reason: str) -> str | None:
    m = re.search(r"Observation:\s*(.+?)(?:\n|$)", reason or "")
    return m.group(1).strip() if m else None


def items(raw: Path, shard: int, shards: int):
    import h5py
    from PIL import Image

    raw = Path(raw)
    root = raw / "failtime" / "failtime_short"
    rows = [(r, False) for r in json.load(open(root / "train.json"))] + \
           [(r, True) for r in json.load(open(root / "test.json"))]
    kinds = sorted({t for r, _ in rows for t in (r.get("type") or [])})
    pool: dict[str, list[str]] = {}
    for r, _ in rows:
        obs = _observed(r.get("reason", ""))
        if obs:
            pool.setdefault(r["task_id"], []).append(obs)
    for n, (r, test) in enumerate(rows):
        if n % shards != shard:
            continue
        path = raw / "vifailback" / "raw_data" / r["hdf5"].split("episodes/", 1)[1]
        task_xml = root / "tasks" / f"{r['task_id']}.xml"
        if not path.exists() or not task_xml.exists():
            continue
        task = ET.parse(task_xml).getroot().findtext("description", "").strip()
        with h5py.File(path) as f:
            frames = len(f["observations/images/cam_high"])
            cams = [c for c in CAMERAS if f"observations/images/{c}" in f]

        def read(camera: str, ts: list[float], path=path, frames=frames) -> dict:
            out = {}
            with h5py.File(path) as f:
                ds = f[f"observations/images/{camera}"]
                for t in ts:
                    k = min(frames - 1, max(0, round(t * FPS)))
                    out[t] = Image.open(io.BytesIO(bytes(ds[k]).rstrip(b"\0"))).convert("RGB")
            return out

        fail = r["verdict"] == "fail"
        types = r.get("type") or []
        obs = _observed(r.get("reason", ""))
        yield Episode(source="failtime", episode_id=r["sample"], family=f"failtime:{r['task_id']}:{r['episode']}",
                      task=task, duration=frames / FPS, cameras=cams,
                      frame=lambda c, t, read=read: read(c, [t])[t], frames=read,
                      robot="AgileX Piper dual-arm", success=not fail, success_from="human",
                      failure_onset=float(r["onset"]) if fail and r.get("onset") is not None else None,
                      failure_type=types[0] if fail and len(types) == 1 else None, failure_types=kinds,
                      failure_from="human", designated_test=test,
                      extra={"failure_reason": obs if fail else None,
                             "reason_pool": [o for o in pool.get(r["task_id"], []) if o != obs]})
