"""Frozen SigLIP patch/text features, content-addressed and reusable across heads."""
from __future__ import annotations

import json
import io
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from .data import DownloadBudget, RR_REPO, RR_REVISION, atomic_json, digest, file_digest, read_manifest

ENCODER_ID = "google/siglip2-base-patch16-224"
ENCODER_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
FRAME_COUNT = 8


def sample_video(path: Path | bytes, frame_count: int = FRAME_COUNT) -> tuple[list, Tensor]:
    """Decode by presentation time; choose nearest uniform timestamps including endpoints."""
    import av
    if frame_count < 2:
        raise ValueError("Need at least two observations")
    times: list[float] = []
    with av.open(io.BytesIO(path) if isinstance(path, bytes) else str(path)) as container:
        for frame in container.decode(video=0):
            if frame.time is None:
                raise ValueError("Video lacks presentation timestamps")
            times.append(float(frame.time))
            if len(times) > 100_000:
                raise ValueError("Video exceeds bounded gate decode limit")
    if not times or times != sorted(times):
        raise ValueError("Empty video or nonmonotonic presentation timestamps")
    targets = torch.linspace(times[0], times[-1], frame_count).tolist()
    indices = [min(range(len(times)), key=lambda i: abs(times[i] - t)) for t in targets]
    wanted = set(indices)
    decoded = {}
    with av.open(io.BytesIO(path) if isinstance(path, bytes) else str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                decoded[i] = frame.to_image()
            if i >= max(wanted):
                break
    if set(decoded) != wanted:
        raise ValueError("Video changed between timestamp and image decode")
    return [decoded[i] for i in indices], torch.tensor([times[i] for i in indices], dtype=torch.float32)


class FrozenSiglip:
    def __init__(self, device: str = "cpu", local_files_only: bool = False):
        import transformers
        from transformers import AutoModel, AutoProcessor
        self.device = torch.device(device)
        if not local_files_only:
            from huggingface_hub import try_to_load_from_cache
            from huggingface_hub.constants import HF_HUB_CACHE
            weights = try_to_load_from_cache(ENCODER_ID, "model.safetensors", revision=ENCODER_REVISION)
            hub_root = Path(HF_HUB_CACHE)
            hub_root.mkdir(parents=True, exist_ok=True)
            # Pinned checkpoint is 1,500,800,904 bytes; tokenizer/config add <40 MB.
            required = (0 if isinstance(weights, str) else 1_500_800_904) + 40_000_000 + 512 * 1024**2
            if shutil.disk_usage(hub_root).free < required:
                raise ValueError("Insufficient disk for pinned encoder download plus reserve")
        self.processor = AutoProcessor.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION,
                                                       local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(ENCODER_ID, revision=ENCODER_REVISION,
                                               local_files_only=local_files_only).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        config = self.model.config
        if (config.vision_config.hidden_size, config.vision_config.image_size, config.vision_config.patch_size) != (768, 224, 16):
            raise ValueError("Pinned encoder geometry changed")
        self.spec = {"schema": 1, "model": ENCODER_ID, "revision": ENCODER_REVISION,
                     "processor": self.processor.image_processor.to_dict(),
                     "text_max_tokens": 64, "text_overlength": "reject", "frames": FRAME_COUNT,
                     "sampling": "nearest-uniform-presentation-time-including-endpoints-v1",
                     "cache_dtype": "bfloat16", "forward_dtype": "float32", "device_type": self.device.type,
                     "torch_version": torch.__version__, "transformers_version": transformers.__version__}
        self.encoder_digest = digest(self.spec)

    @torch.inference_mode()
    def images(self, images: list, batch_size: int = 4) -> Tensor:
        self.model.eval()
        pieces = []
        for offset in range(0, len(images), batch_size):
            batch = self.processor(images=images[offset:offset + batch_size], return_tensors="pt")
            output = self.model.vision_model(pixel_values=batch["pixel_values"].to(self.device))
            pieces.append(output.last_hidden_state.detach().to(device="cpu", dtype=torch.bfloat16))
        result = torch.cat(pieces)
        if result.shape[1:] != (196, 768) or not torch.isfinite(result).all():
            raise ValueError("Invalid frozen visual features")
        return result

    @torch.inference_mode()
    def text(self, task: str) -> tuple[Tensor, Tensor]:
        tokens = self.processor.tokenizer(task, add_special_tokens=True, truncation=False)
        if len(tokens["input_ids"]) > 64:
            raise ValueError("Instruction exceeds 64-token budget; fix manifest explicitly, no silent truncation")
        inputs = self.processor.tokenizer(task, padding="max_length", max_length=64, truncation=False,
                                          return_attention_mask=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
        self.model.eval()
        out = self.model.text_model(**inputs).last_hidden_state[0]
        if not torch.isfinite(out).all():
            raise ValueError("Invalid frozen text features")
        return out.to(device="cpu", dtype=torch.bfloat16), inputs["attention_mask"][0].to(device="cpu", dtype=torch.bool)


class FeatureCache:
    """Single-writer cache with optional lossless byte-shuffled zlib storage.

    Compression changes storage only. The encoded envelope contains the exact
    safetensors bytes, their original length and SHA256. Both representations
    remain readable irrespective of the constructor's write preference.
    """
    _MAGIC = b"SXFSZ01\x00"
    _MAX_ENTRY_BYTES = 256 * 1024**2

    def __init__(self, root: Path, max_bytes: int = 512 * 1024**2,
                 reserve_bytes: int = 512 * 1024**2, compressed: bool = False):
        self.root, self.max_bytes, self.reserve_bytes = root, max_bytes, reserve_bytes
        self.compressed = compressed
        if max_bytes <= 0 or reserve_bytes < 0:
            raise ValueError("Invalid feature cache budget or disk reserve")
        root.mkdir(parents=True, exist_ok=True)
        # One scan on construction; feature production must not rescan thousands
        # of previous entries for each new episode.
        self.current_bytes = sum(
            path.stat().st_size for path in root.iterdir()
            if path.is_file() and path.name.endswith((".safetensors", ".safetensors.sz"))
        )

    def path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid cache key")
        plain = self.root / (key + ".safetensors")
        packed = self.root / (key + ".safetensors.sz")
        if plain.exists():
            if packed.exists():
                raise ValueError("Ambiguous duplicate feature cache representations")
            return plain
        if packed.exists():
            return packed
        return packed if self.compressed else plain

    @classmethod
    def _pack(cls, raw: bytes) -> bytes:
        import hashlib
        import struct
        import zlib
        if len(raw) > cls._MAX_ENTRY_BYTES:
            raise ValueError("Feature entry exceeds bounded codec size")
        # Separating low/high byte planes improves BF16 compression while also
        # preserving arbitrary headers, FP32 timestamps and boolean masks.
        shuffled = raw[0::2] + raw[1::2]
        return (cls._MAGIC + struct.pack("<Q", len(raw)) + hashlib.sha256(raw).digest()
                + zlib.compress(shuffled, level=1))

    @classmethod
    def _unpack(cls, packed: bytes) -> bytes:
        import hashlib
        import struct
        import zlib
        if len(packed) < 48 or packed[:8] != cls._MAGIC:
            raise ValueError("Invalid compressed feature cache envelope")
        size = struct.unpack("<Q", packed[8:16])[0]
        if size > cls._MAX_ENTRY_BYTES:
            raise ValueError("Feature entry exceeds bounded codec size")
        try:
            decoder = zlib.decompressobj()
            shuffled = decoder.decompress(packed[48:], max_length=size + 1)
        except zlib.error as exc:
            raise ValueError("Compressed feature bytes are corrupt") from exc
        if (len(shuffled) != size or not decoder.eof
                or decoder.unused_data or decoder.unconsumed_tail):
            raise ValueError("Compressed feature length or stream mismatch")
        middle = (size + 1) // 2
        raw = bytearray(size)
        raw[0::2], raw[1::2] = shuffled[:middle], shuffled[middle:]
        if hashlib.sha256(raw).digest() != packed[16:48]:
            raise ValueError("Compressed feature SHA256 mismatch")
        return bytes(raw)

    def read(self, key: str) -> dict[str, Tensor]:
        import struct
        from safetensors.torch import load
        path = self.path(key)
        raw = path.read_bytes()
        if path.name.endswith(".sz"):
            raw = self._unpack(raw)
        # safetensors validates every tensor's dtype, shape, offsets and payload;
        # inspect the same validated bytes for their content identity.
        tensors = load(raw)
        header_length = struct.unpack("<Q", raw[:8])[0]
        meta = json.loads(raw[8:8 + header_length]).get("__metadata__")
        if not meta or meta.get("key") != key or digest(json.loads(meta["identity"])) != key:
            raise ValueError("Feature cache identity mismatch")
        for value in tensors.values():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Nonfinite feature cache entry")
        return tensors

    def put(self, identity: dict, tensors: dict[str, Tensor]) -> str:
        from safetensors.torch import save
        key = digest(identity)
        dest = self.path(key)
        if dest.exists():
            self.read(key)
            return key
        for value in tensors.values():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Refusing nonfinite cache entry")
        raw = save({k: v.detach().cpu().contiguous() for k, v in tensors.items()},
                   metadata={"key": key, "identity": json.dumps(identity, sort_keys=True)})
        payload = self._pack(raw) if self.compressed else raw
        size = len(payload)
        if self.current_bytes + size > self.max_bytes:
            raise ValueError("Feature cache byte budget exhausted")
        if shutil.disk_usage(self.root).free < size + self.reserve_bytes:
            raise ValueError("Insufficient disk for feature cache plus reserve")
        tmp = dest.with_name(dest.name + ".tmp")
        with tmp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, dest)
        self.current_bytes += size
        return key


@dataclass(frozen=True)
class FeatureRow:
    record_id: str
    visual_key: str
    text_key: str
    media_sha256: str
    visual_sha256: str
    text_sha256: str
    media_bytes: int = 0


def build_cache(manifest: Path, output: Path, encoder: FrozenSiglip, media_budget_mb: int,
                feature_budget_mb: int, *, stream_media: bool = False,
                compressed: bool = False, reuse: Path | None = None) -> dict:
    """Durably write each paid feature before advancing; resume compatible partial indexes."""
    from dataclasses import asdict
    records, manifest_digest = read_manifest(manifest)
    cache = FeatureCache(output / "features", max_bytes=feature_budget_mb * 1024**2, compressed=compressed)
    from .streaming import StreamingMedia
    media = StreamingMedia(records, output / "media-inventory.json", media_budget_mb * 1024**2) if stream_media else None
    old_cache = FeatureCache(reuse / "features") if reuse else None
    reuse_hashes: dict[str, str] = {}
    if reuse:
        old_index = json.loads((reuse / "index.json").read_text())
        if (old_index["encoder_digest"] != encoder.encoder_digest
                or digest(old_index["encoder"]) != encoder.encoder_digest):
            raise ValueError("Reuse cache encoder differs from current frozen encoder")
        for row in old_index["rows"]:
            for field in ("visual", "text"):
                key, sha = row[field + "_key"], row[field + "_sha256"]
                if key in reuse_hashes and reuse_hashes[key] != sha:
                    raise ValueError("Reuse index has conflicting feature hashes")
                reuse_hashes[key] = sha

    def can_reuse(key: str) -> bool:
        return old_cache is not None and key in reuse_hashes and old_cache.path(key).exists()

    def reuse_feature(identity: dict, key: str) -> None:
        if old_cache is None or file_digest(old_cache.path(key)) != reuse_hashes[key]:
            raise ValueError("Reuse feature bytes differ from original index")
        cache.put(identity, old_cache.read(key))
    # Reject overlength instructions before paying for any image encoding.
    for task in {r.task for r in records}:
        if len(encoder.processor.tokenizer(task, add_special_tokens=True, truncation=False)["input_ids"]) > 64:
            raise ValueError("Instruction exceeds frozen 64-token budget")
    budget = DownloadBudget(media_budget_mb * 1024**2)
    index_path = output / "index.json"
    prior = json.loads(index_path.read_text()) if index_path.exists() else None
    if prior and (prior["manifest_digest"] != manifest_digest or prior["encoder_digest"] != encoder.encoder_digest):
        raise ValueError("Refusing to reuse a cache index for different inputs/encoder")
    rows = {r["record_id"]: r for r in prior["rows"]} if prior else {}
    by_hash: dict[str, str] = {}
    for r in records:
        if r.id in rows:
            old = rows[r.id]
            size = old.get("media_bytes", 0)
            if size <= 0:
                raise ValueError("Partial cache lacks durable media size accounting")
            budget.seen[f"{RR_REPO}@{RR_REVISION}/{r.video}"] = size
            if sum(budget.seen.values()) > budget.max_bytes:
                raise ValueError("Existing selected media exceeds cumulative byte budget")
            for key, expected in ((old["visual_key"], old["visual_sha256"]), (old["text_key"], old["text_sha256"])):
                if file_digest(cache.path(key)) != expected:
                    raise ValueError("Existing cache bytes changed")
            sha = old["media_sha256"]
            if sha in by_hash and by_hash[sha] != r.split:
                raise ValueError("Exact video bytes cross train/dev split")
            by_hash[sha] = r.split
    started = time.monotonic()
    encoded = 0
    pending_paths, pending_hashes = [], set()
    if media:
        for r in records:
            sha = media.inventory[r.video]["sha256"]
            key = digest({"encoder": encoder.encoder_digest, "kind": "visual", "media_sha256": sha})
            if sha not in pending_hashes and not cache.path(key).exists() and not can_reuse(key):
                pending_paths.append(r.video)
                pending_hashes.add(sha)
    streamed = media.iter_verified(pending_paths) if media else None
    for r in records:
        if r.id in rows:
            continue
        video = None if media else budget.download(RR_REPO, RR_REVISION, r.video)
        sha = media.inventory[r.video]["sha256"] if media else file_digest(video)
        media_size = media.inventory[r.video]["size"] if media else video.stat().st_size
        if sha in by_hash and by_hash[sha] != r.split:
            raise ValueError("Exact video bytes cross train/dev split; quarantine before training")
        by_hash[sha] = r.split
        visual_identity = {"encoder": encoder.encoder_digest, "kind": "visual", "media_sha256": sha}
        vkey = digest(visual_identity)
        if can_reuse(vkey) and not cache.path(vkey).exists():
            reuse_feature(visual_identity, vkey)
        if not cache.path(vkey).exists():
            if media:
                filename, video = next(streamed)
                if filename != r.video:
                    raise ValueError("Prefetched media order differs from encoding membership")
            images, seconds = sample_video(video)
            visual = encoder.images(images).unsqueeze(1)  # T,V,P,D (single RR camera)
            duration = seconds[-1] - seconds[0]
            times = (seconds - seconds[0]) / duration if duration > 0 else torch.zeros_like(seconds)
            cache.put(visual_identity, {"visual": visual, "times": times, "seconds": seconds,
                                         "view_mask": torch.ones((FRAME_COUNT, 1), dtype=torch.bool)})
            encoded += 1
        else:
            cache.read(vkey)
        text_identity = {"encoder": encoder.encoder_digest, "kind": "text", "task": r.task}
        tkey = digest(text_identity)
        if can_reuse(tkey) and not cache.path(tkey).exists():
            reuse_feature(text_identity, tkey)
        if not cache.path(tkey).exists():
            text, mask = encoder.text(r.task)
            cache.put(text_identity, {"text": text, "text_mask": mask})
        else:
            cache.read(tkey)
        row = FeatureRow(r.id, vkey, tkey, sha, file_digest(cache.path(vkey)), file_digest(cache.path(tkey)), media_size)
        rows[r.id] = asdict(row)
        index = {"schema": 1, "manifest_digest": manifest_digest, "encoder_digest": encoder.encoder_digest,
                 "encoder": encoder.spec, "complete": len(rows) == len(records),
                 "rows": [rows[x.id] for x in records if x.id in rows]}
        atomic_json(index_path, index)
        print(json.dumps({"cached": len(rows), "total": len(records), "new_videos": encoded,
                          "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    if not records:
        raise ValueError("Empty manifest")
    return json.loads(index_path.read_text())


def load_features(manifest: Path, cache_dir: Path) -> tuple[list, dict[str, dict[str, Tensor]], str]:
    records, manifest_digest = read_manifest(manifest)
    index = json.loads((cache_dir / "index.json").read_text())
    if digest(index["encoder"]) != index["encoder_digest"]:
        raise ValueError("Encoder specification digest mismatch")
    if not index["complete"] or index["manifest_digest"] != manifest_digest:
        raise ValueError("Incomplete or incompatible feature cache")
    rows = {row["record_id"]: FeatureRow(**row) for row in index["rows"]}
    if set(rows) != {r.id for r in records}:
        raise ValueError("Feature cache membership differs from manifest")
    cache = FeatureCache(cache_dir / "features")
    tensors: dict[str, dict[str, Tensor]] = {}
    entries = {}
    media_membership: dict[str, str] = {}
    for r in records:
        row = rows[r.id]
        if row.text_key != digest({"encoder": index["encoder_digest"], "kind": "text", "task": r.task}):
            raise ValueError("Text cache is bound to a different instruction")
        if row.visual_key != digest({"encoder": index["encoder_digest"], "kind": "visual", "media_sha256": row.media_sha256}):
            raise ValueError("Visual cache is bound to different media")
        if row.media_sha256 in media_membership and media_membership[row.media_sha256] != r.split:
            raise ValueError("Exact media duplicated across train/dev")
        media_membership[row.media_sha256] = r.split
        for key, sha in ((row.visual_key, row.visual_sha256), (row.text_key, row.text_sha256)):
            if key not in tensors:
                if file_digest(cache.path(key)) != sha:
                    raise ValueError("Feature bytes differ from durable index")
                tensors[key] = cache.read(key)
        entries[r.id] = {**tensors[row.visual_key], **tensors[row.text_key]}
    return records, entries, digest(index)
