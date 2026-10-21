"""Frozen SigLIP patch/text features, content-addressed and reusable across heads."""
from __future__ import annotations

import json
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


def sample_video(path: Path, frame_count: int = FRAME_COUNT) -> tuple[list, Tensor]:
    """Decode by presentation time; choose nearest uniform timestamps including endpoints."""
    import av
    if frame_count < 2:
        raise ValueError("Need at least two observations")
    times: list[float] = []
    with av.open(str(path)) as container:
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
    with av.open(str(path)) as container:
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
    def __init__(self, root: Path, max_bytes: int = 512 * 1024**2, reserve_bytes: int = 512 * 1024**2):
        self.root, self.max_bytes, self.reserve_bytes = root, max_bytes, reserve_bytes
        root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid cache key")
        return self.root / (key + ".safetensors")

    def read(self, key: str) -> dict[str, Tensor]:
        from safetensors import safe_open
        from safetensors.torch import load_file
        path = self.path(key)
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata()
            if not meta or meta.get("key") != key or digest(json.loads(meta["identity"])) != key:
                raise ValueError("Feature cache identity mismatch")
        return load_file(str(path), device="cpu")

    def put(self, identity: dict, tensors: dict[str, Tensor]) -> str:
        from safetensors.torch import save_file
        key = digest(identity)
        dest = self.path(key)
        if dest.exists():
            self.read(key)
            return key
        projected = sum(t.numel() * t.element_size() for t in tensors.values()) + 16_384
        current = sum(p.stat().st_size for p in self.root.glob("*.safetensors"))
        if current + projected > self.max_bytes:
            raise ValueError("Feature cache byte budget exhausted")
        if shutil.disk_usage(self.root).free < projected + self.reserve_bytes:
            raise ValueError("Insufficient disk for feature cache plus reserve")
        for value in tensors.values():
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError("Refusing nonfinite cache entry")
        tmp = dest.with_suffix(".tmp")
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp),
                  metadata={"key": key, "identity": json.dumps(identity, sort_keys=True)})
        os.replace(tmp, dest)
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
                feature_budget_mb: int) -> dict:
    """Durably write each paid feature before advancing; resume compatible partial indexes."""
    from dataclasses import asdict
    records, manifest_digest = read_manifest(manifest)
    cache = FeatureCache(output / "features", max_bytes=feature_budget_mb * 1024**2)
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
    for r in records:
        if r.id in rows:
            continue
        video = budget.download(RR_REPO, RR_REVISION, r.video)
        sha = file_digest(video)
        if sha in by_hash and by_hash[sha] != r.split:
            raise ValueError("Exact video bytes cross train/dev split; quarantine before training")
        by_hash[sha] = r.split
        visual_identity = {"encoder": encoder.encoder_digest, "kind": "visual", "media_sha256": sha}
        vkey = digest(visual_identity)
        if not cache.path(vkey).exists():
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
        if not cache.path(tkey).exists():
            text, mask = encoder.text(r.task)
            cache.put(text_identity, {"text": text, "text_mask": mask})
        else:
            cache.read(tkey)
        row = FeatureRow(r.id, vkey, tkey, sha, file_digest(cache.path(vkey)), file_digest(cache.path(tkey)), video.stat().st_size)
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
