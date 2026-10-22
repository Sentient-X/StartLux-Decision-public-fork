"""Frames out of episode videos by time, with one open decoder per file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class VideoFrames:
    """frame(t) -> the PIL image shown at t seconds after `offset` in an mp4 (LeRobot v3 files hold several episodes)."""

    def __init__(self, path: Path, offset: float = 0.0):
        self.path, self.offset = Path(path), offset
        self._c = None

    def _container(self):
        import av
        if self._c is None:
            self._c = av.open(str(self.path))
            self._c.streams.video[0].thread_type = "AUTO"
        return self._c

    def frame(self, t: float):
        c = self._container()
        s = c.streams.video[0]
        if s.time_base is None:
            raise ValueError(f"{self.path}: video stream has no time base")
        target = self.offset + max(0.0, t)
        c.seek(int(max(0.0, target - 0.5) / s.time_base), stream=s, backward=True, any_frame=False)
        last = None
        for f in c.decode(s):
            if f.time is None:
                continue
            last = f
            if f.time >= target - 0.5 / max(float(s.average_rate or 30), 1):
                break
        if last is None:
            raise ValueError(f"no frame at {t:.2f}s in {self.path}")
        return last.to_image()

    def close(self):
        if self._c is not None:
            self._c.close()
            self._c = None


@lru_cache(maxsize=8)
def opened(path: str, offset: float) -> VideoFrames:
    return VideoFrames(Path(path), offset)
