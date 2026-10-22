"""Pinned, size-bounded RoboReward downloads retained only in RAM."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import RR_REPO, RR_REVISION, Record, atomic_json, digest


class StreamingMedia:
    def __init__(self, records: list[Record], inventory_path: Path, max_bytes: int):
        from huggingface_hub import HfApi
        paths = {r.video for r in records}
        if inventory_path.exists():
            payload = json.loads(inventory_path.read_text())
            claimed = payload.pop("digest")
            if digest(payload) != claimed or payload["repo"] != RR_REPO or payload["revision"] != RR_REVISION:
                raise ValueError("Media inventory identity mismatch")
            inventory = payload["files"]
        else:
            info = HfApi().dataset_info(RR_REPO, revision=RR_REVISION, files_metadata=True)
            inventory = {f.rfilename: {"size": f.size, "sha256": f.lfs.sha256 if f.lfs else None}
                         for f in info.siblings if f.rfilename in paths}
            payload = {"repo": RR_REPO, "revision": RR_REVISION, "files": inventory}
            atomic_json(inventory_path, {**payload, "digest": digest(payload)})
        if set(inventory) != paths:
            raise ValueError("Media inventory membership mismatch")
        memberships: dict[str, str] = {}
        for r in records:
            item = inventory[r.video]
            size, sha = item["size"], item["sha256"]
            if type(size) is not int or not 0 < size <= 256 * 1024**2 or not isinstance(sha, str) or len(sha) != 64:
                raise ValueError("Media inventory lacks bounded size/content identity")
            if sha in memberships and memberships[sha] != r.split:
                raise ValueError("Exact video bytes cross train/dev split")
            memberships[sha] = r.split
        # Count all selected paths conservatively, including content duplicates.
        if sum(item["size"] for item in inventory.values()) > max_bytes:
            raise ValueError("Selected media exceeds cumulative byte budget")
        self.inventory = inventory

    def iter_verified(self, filenames: list[str]):
        """Overlap bounded network reads with encoding; at most eight media in RAM."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        remaining = iter(filenames)
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = deque()
            for _ in range(8):
                name = next(remaining, None)
                if name is not None:
                    pending.append((name, pool.submit(self.read, name)))
            while pending:
                name, future = pending.popleft()
                data = future.result()
                following = next(remaining, None)
                if following is not None:
                    pending.append((following, pool.submit(self.read, following)))
                yield name, data

    def read(self, filename: str) -> bytes:
        import httpx
        from huggingface_hub import hf_hub_url, try_to_load_from_cache
        item = self.inventory[filename]
        cached = try_to_load_from_cache(RR_REPO, filename, repo_type="dataset", revision=RR_REVISION)
        if isinstance(cached, str):
            data = Path(cached).read_bytes()
        else:
            url = hf_hub_url(RR_REPO, filename, repo_type="dataset", revision=RR_REVISION)
            try:
                with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
                    if response.status_code != 200:
                        raise ValueError(f"Media HTTP status {response.status_code} for {filename}")
                    chunks, total = [], 0
                    for chunk in response.iter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > item["size"]:
                            raise ValueError("Stream exceeded pinned media size")
                        chunks.append(chunk)
                    data = b"".join(chunks)
            except httpx.HTTPError as exc:
                # Do not include signed redirect URLs in logs.
                raise RuntimeError(f"Media transport failed for {filename}: {type(exc).__name__}") from None
        if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Streamed media differs from pinned size/content identity")
        return data
