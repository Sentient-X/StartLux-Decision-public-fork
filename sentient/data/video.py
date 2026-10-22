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

    def frames(self, times: list[float]) -> dict[float, "object"]:
        """{t: PIL image} for many times at once: one sequential decode, keeping the first frame at or after each t."""
        import av
        want = sorted(set(times))
        out: dict[float, object] = {}
        if not want:
            return out
        with av.open(str(self.path)) as c:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            half = 0.5 / max(float(s.average_rate or 30), 1)
            start = self.offset + want[0]
            if s.time_base is not None and start > 2:
                c.seek(int((start - 1) / s.time_base), stream=s, backward=True, any_frame=False)
            i, last = 0, None
            for f in c.decode(s):
                if f.time is None:
                    continue
                last = f
                while i < len(want) and f.time >= self.offset + want[i] - half:
                    out[want[i]] = f.to_image()
                    i += 1
                if i == len(want) or f.time > self.offset + want[-1] + 1:
                    break
            while i < len(want) and last is not None:      # times past the last frame get the last frame
                out[want[i]] = last.to_image()
                i += 1
        return out

    def close(self):
        if self._c is not None:
            self._c.close()
            self._c = None


@lru_cache(maxsize=8)
def opened(path: str, offset: float) -> VideoFrames:
    return VideoFrames(Path(path), offset)
