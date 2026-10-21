"""Pinned RoboReward membership and bounded media acquisition (no test labels)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

RR_REPO = "teetone/RoboReward"
RR_REVISION = "469b9af76e8539e8d2ac553b081307738fd92ca5"
Split = Literal["train", "dev"]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def family_of(filename: str) -> str:
    """Conservative known lineage: OXE original index or RoboArena recording UUID.

    Fail closed on unrecognized names instead of treating unknown clips as independent.
    """
    p = PurePosixPath(filename)
    if p.is_absolute() or ".." in p.parts or len(p.parts) != 2 or p.suffix != ".mp4":
        raise ValueError(f"Invalid RoboReward relative video path: {filename}")
    oxe = re.match(r"(.+_originalsplit_[A-Za-z0-9_]+_index_\d+)(?:_.*)?$", p.stem)
    arena = re.match(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})(?:_.*)?$", p.stem)
    if p.parts[0] == "robo_arena" and arena:
        return f"robo_arena/{arena[1]}"
    if oxe:
        return f"{p.parts[0]}/{oxe[1]}"
    raise ValueError(f"Unknown episode lineage: {filename}")


@dataclass(frozen=True)
class Record:
    id: str
    family: str
    split: Split
    video: str
    task: str
    reward: int

    @classmethod
    def parse(cls, row: dict, split: Split, official_split: str) -> Record:
        filename, task, reward = row.get("file_name"), row.get("task"), row.get("reward")
        if not isinstance(filename, str) or not isinstance(task, str) or not task.strip():
            raise ValueError("RoboReward row requires file_name and nonempty task")
        if type(reward) is not int or reward not in range(1, 6):
            raise ValueError("RoboReward reward must be integer 1..5")
        if official_split not in ("train", "val"):
            raise ValueError("Gate must not load official test labels")
        family = family_of(filename)
        rid = digest([RR_REPO, RR_REVISION, official_split, filename, task])
        return cls(rid, family, split, f"{official_split}/{filename}", task.strip(), reward)


def choose_groups(records: list[Record], count: int, seed: int) -> list[Record]:
    if count <= 0:
        raise ValueError("Requested record count must be positive")
    groups: dict[str, list[Record]] = {}
    for r in records:
        groups.setdefault(r.family, []).append(r)
    result: list[Record] = []
    for family in sorted(groups, key=lambda k: digest([seed, k])):
        result.extend(sorted(groups[family], key=lambda r: r.id))
        if len(result) >= count:
            return result
    raise ValueError(f"Only {len(result)} eligible records for requested {count}")


def validate_records(records: list[Record]) -> None:
    membership: dict[str, str] = {}
    ids: set[str] = set()
    for r in records:
        if r.split not in ("train", "dev") or r.video.split("/")[0] != {"train": "train", "dev": "val"}[r.split]:
            raise ValueError("Invalid membership/source split")
        if r.family != family_of(r.video.split("/", 1)[1]):
            raise ValueError("Manifest lineage does not match video")
        if r.id in ids:
            raise ValueError("Duplicate record ID")
        ids.add(r.id)
        if r.family in membership and membership[r.family] != r.split:
            raise ValueError("Episode family crosses training and held-out membership")
        membership[r.family] = r.split
        if type(r.reward) is not int or r.reward not in range(1, 6) or not r.task.strip():
            raise ValueError("Invalid target or task")


def read_manifest(path: Path) -> tuple[list[Record], str]:
    data = json.loads(path.read_text())
    claimed = data.pop("digest")
    if digest(data) != claimed or data["repo"] != RR_REPO or data["revision"] != RR_REVISION:
        raise ValueError("Manifest digest/revision mismatch")
    records = [Record(**row) for row in data["records"]]
    validate_records(records)
    return records, claimed


def prepare(path: Path, train_count: int, dev_count: int, seed: int = 17) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    if path.exists():
        read_manifest(path)
        existing = json.loads(path.read_text())
        if existing["requested"] != {"train": train_count, "dev": dev_count} or existing["seed"] != seed:
            raise ValueError("Existing immutable manifest has different selection settings; use a new path")
        return existing
    info = HfApi().dataset_info(RR_REPO, revision=RR_REVISION)
    # Reserve all official validation/test families before selection, using filenames only.
    # Test metadata, instructions and rewards are never downloaded.
    reserved: set[str] = set()
    test_families: set[str] = set()
    for sibling in info.siblings:
        split, _, filename = sibling.rfilename.partition("/")
        if split in ("val", "test") and filename.endswith(".mp4"):
            reserved.add(family_of(filename))
            if split == "test":
                test_families.add(family_of(filename))
    selected: list[Record] = []
    exclusion_count = 0
    metadata_hashes: dict[str, str] = {}
    for official, role, count in (("train", "train", train_count), ("val", "dev", dev_count)):
        meta = Path(hf_hub_download(RR_REPO, f"{official}/metadata.jsonl", repo_type="dataset", revision=RR_REVISION))
        metadata_hashes[official] = file_digest(meta)
        rows = [Record.parse(json.loads(line), role, official) for line in meta.read_text().splitlines() if line.strip()]
        if role == "train":
            before = len(rows)
            rows = [r for r in rows if r.family not in reserved]
            exclusion_count = before - len(rows)
        else:
            rows = [r for r in rows if r.family not in test_families]
        selected.extend(choose_groups(rows, count, seed))
    validate_records(selected)
    result = {"version": 1, "repo": RR_REPO, "revision": RR_REVISION, "seed": seed,
              "requested": {"train": train_count, "dev": dev_count},
              "train_rows_quarantined": exclusion_count, "reserved_family_count": len(reserved),
              "metadata_sha256": metadata_hashes,
              "overlap_scope": "known source lineage; exact downloaded bytes checked again before training; no full-corpus perceptual dedup claim",
              "records": [asdict(r) for r in selected]}
    result["digest"] = digest(result)
    atomic_json(path, result)
    return result


class DownloadBudget:
    """Cumulative unique selected-media budget, including files already cached."""
    def __init__(self, max_bytes: int, reserve_bytes: int = 512 * 1024**2):
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes
        self.seen: dict[str, int] = {}

    def download(self, repo: str, revision: str, filename: str) -> Path:
        from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url, try_to_load_from_cache
        key = f"{repo}@{revision}/{filename}"
        cached = try_to_load_from_cache(repo, filename, repo_type="dataset", revision=revision)
        local = Path(cached) if isinstance(cached, str) else None
        size = local.stat().st_size if local else get_hf_file_metadata(hf_hub_url(repo, filename, repo_type="dataset", revision=revision)).size
        if size is None or size <= 0:
            raise ValueError("Media size unavailable; refusing an unbounded download")
        if key not in self.seen and sum(self.seen.values()) + size > self.max_bytes:
            raise ValueError(f"Selected media exceeds {self.max_bytes} byte budget before {filename}")
        if local is None:
            from huggingface_hub.constants import HF_HUB_CACHE
            root = Path(HF_HUB_CACHE)
            root.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(root).free < size + self.reserve_bytes:
                raise ValueError("Insufficient disk for media plus reserve")
            local = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
        if local.stat().st_size != size:
            raise ValueError("Downloaded size mismatch")
        self.seen[key] = size
        return local
