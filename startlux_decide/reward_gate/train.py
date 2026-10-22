"""Three matched heads, persistent optimizer/RNG state, and development-only gate."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .data import Record, atomic_json, digest, file_digest
from .features import load_features
from .heads import HEAD_KINDS, HeadConfig, HeadKind, make_head, outcome_loss


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    microbatch: int = 8
    effective_batch: int = 64
    learning_rate: float = 3e-4
    seed: int = 17
    device: str = "cpu"

    def __post_init__(self):
        if min(self.epochs, self.microbatch, self.effective_batch) <= 0 or self.learning_rate <= 0:
            raise ValueError("Training settings must be positive")
        if self.effective_batch % self.microbatch:
            raise ValueError("Effective batch must be a multiple of microbatch")


def batch_of(records: Sequence[Record], entries: dict, device: str) -> tuple[dict[str, Tensor], Tensor]:
    keys = ("visual", "text", "text_mask", "view_mask", "times")
    batch = {k: torch.stack([entries[r.id][k] for r in records]).to(device) for k in keys}
    # Keep all non-padding tokens, without paying attention cost for fixed padding.
    last = int(batch["text_mask"].any(dim=0).nonzero()[-1].item()) + 1
    batch["text"], batch["text_mask"] = batch["text"][:, :last], batch["text_mask"][:, :last]
    targets = torch.tensor([r.reward - 1 for r in records], device=device, dtype=torch.long)
    return batch, targets


def capture_rng() -> dict:
    state = {"cpu": torch.get_rng_state(), "python": random.getstate()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    random.setstate(state["python"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@torch.inference_mode()
def evaluate(model, records: list[Record], entries: dict, cfg: TrainConfig) -> tuple[dict, list[dict]]:
    model.eval()
    predictions = []
    for offset in range(0, len(records), cfg.microbatch):
        rows = records[offset:offset + cfg.microbatch]
        batch, targets = batch_of(rows, entries, cfg.device)
        probabilities = model(**batch).softmax(-1).cpu()
        if not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite evaluation predictions")
        for r, p in zip(rows, probabilities):
            predictions.append({"id": r.id, "family": r.family, "source": r.family.split('/')[0],
                                "gold": r.reward, "prediction": int(p.argmax()) + 1,
                                "probabilities": p.tolist()})
    if not predictions:
        raise ValueError("Empty development set")
    mae = sum(abs(p["prediction"] - p["gold"]) for p in predictions) / len(predictions)
    brier = sum(sum((v - int(i == p["gold"] - 1)) ** 2 for i, v in enumerate(p["probabilities"])) for p in predictions) / len(predictions)
    return {"n": len(predictions), "families": len({p["family"] for p in predictions}),
            "mae": mae, "brier": brier,
            "accuracy": sum(p["gold"] == p["prediction"] for p in predictions) / len(predictions)}, predictions


@torch.inference_mode()
def sensitivity(model, rows: list[Record], entries: dict, cfg: TrainConfig) -> dict:
    """Mechanics check on held-out features; these perturbations are not labels."""
    model.eval()
    batch, _ = batch_of(rows[:max(2, cfg.microbatch)], entries, cfg.device)
    base = model(**batch).softmax(-1)
    changed_text = {**batch, "text": batch["text"].roll(1, 0), "text_mask": batch["text_mask"].roll(1, 0)}
    changed_video = {**batch, **{k: batch[k].roll(1, 0) for k in ("visual", "view_mask", "times")}}
    reversed_frames = {**batch, "visual": batch["visual"].flip(1), "view_mask": batch["view_mask"].flip(1)}
    return {"n": base.shape[0],
            "task_swap_mean_absolute_probability_delta": float((base - model(**changed_text).softmax(-1)).abs().mean()),
            "video_swap_mean_absolute_probability_delta": float((base - model(**changed_video).softmax(-1)).abs().mean()),
            "frame_reversal_mean_absolute_probability_delta": float((base - model(**reversed_frames).softmax(-1)).abs().mean()),
            "interpretation": "input sensitivity only; reversal is not a synthetic failure label"}


def train_one(kind: HeadKind, records: list[Record], entries: dict, output: Path,
              cfg: TrainConfig, cache_digest: str, max_updates: int | None = None,
              head_config: HeadConfig = HeadConfig()) -> dict:
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    train_rows = [r for r in records if r.split == "train"]
    dev_rows = [r for r in records if r.split == "dev"]
    if not train_rows or not dev_rows or {r.family for r in train_rows} & {r.family for r in dev_rows}:
        raise ValueError("Need nonempty disjoint training and development families")
    code = {f.name: file_digest(f) for f in Path(__file__).parent.glob("*.py")}
    identity = {"head": kind, "head_config": asdict(head_config), "train_config": asdict(cfg),
                "cache_digest": cache_digest, "code": code}
    signature = digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, best_path = output / "last.pt", output / "best.pt"
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    model = make_head(kind, head_config).to(cfg.device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters >= 1_000_000:
        raise ValueError("Gate head exceeds one million trainable parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(.9, .999), weight_decay=.01)
    epoch, cursor, updates, best_mae = 0, 0, 0, float("inf")
    history: list[dict] = []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint inputs/config/code differ; refusing silent restart")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        epoch, cursor, updates = saved["epoch"], saved["cursor"], saved["updates"]
        best_mae, history = saved["best_mae"], saved["history"]
        restore_rng(saved["rng"])
    started, updates_this_call = time.monotonic(), 0

    def persist() -> None:
        save_checkpoint(checkpoint, {"signature": signature, "identity": identity,
                                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                    "epoch": epoch, "cursor": cursor, "updates": updates,
                                    "best_mae": best_mae, "history": history, "rng": capture_rng()})

    while epoch < cfg.epochs:
        generator = torch.Generator().manual_seed(cfg.seed + epoch)
        order = torch.randperm(len(train_rows), generator=generator).tolist()
        model.train()
        while cursor < len(order):
            indices = order[cursor:cursor + cfg.effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(indices), cfg.microbatch):
                selected = [train_rows[i] for i in indices[offset:offset + cfg.microbatch]]
                if any(r.split != "train" for r in selected):
                    raise ValueError("Held-out row reached optimizer")
                batch, targets = batch_of(selected, entries, cfg.device)
                loss = outcome_loss(model(**batch), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                (loss * (len(selected) / len(indices))).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            cursor += len(indices)
            updates += 1
            updates_this_call += 1
            persist()  # every completed optimizer group is resumable
            if max_updates is not None and updates_this_call >= max_updates:
                receipt = {"status": "paused_at_update_boundary", "head": kind, "parameters": parameters,
                           "updates": updates, "signature": signature, "elapsed_seconds": time.monotonic() - started}
                atomic_json(output / "receipt.json", receipt)
                return receipt
        metrics, predictions = evaluate(model, dev_rows, entries, cfg)
        atomic_json(output / "last_predictions.json", predictions)
        history.append({"epoch": epoch + 1, **metrics})
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            save_checkpoint(best_path, {"signature": signature, "model": model.state_dict(),
                                         "epoch": epoch + 1, "head_config": asdict(head_config)})
            atomic_json(output / "predictions.json", predictions)
        epoch += 1
        cursor = 0
        persist()
        print(json.dumps({"head": kind, "epoch": epoch, "parameters": parameters, **metrics}), flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    diagnostics = sensitivity(model, dev_rows, entries, cfg)
    receipt = {"status": "complete", "head": kind, "parameters": parameters, "sensitivity": diagnostics, "updates": updates,
               "identity": identity, "signature": signature, "history": history,
               "best_development_mae": best_mae, "elapsed_seconds_this_invocation": time.monotonic() - started,
               "checkpoint_sha256": file_digest(checkpoint), "best_sha256": file_digest(best_path)}
    atomic_json(output / "receipt.json", receipt)
    return receipt


def run(manifest: Path, cache: Path, output: Path, cfg: TrainConfig,
        max_updates: int | None = None) -> dict:
    records, entries, cache_digest = load_features(manifest, cache)
    receipts = [train_one(kind, records, entries, output / kind, cfg, cache_digest, max_updates) for kind in HEAD_KINDS]
    done = all(r["status"] == "complete" for r in receipts)
    report = {"status": "complete" if done else "paused", "purpose": "initial outcome gate; not reference-model superiority",
              "train_records": sum(r.split == "train" for r in records),
              "dev_records": sum(r.split == "dev" for r in records), "heads": receipts,
              "gate_threshold_mae_improvement": .02, "gate_passed": None,
              "eligible_for_outcome_gate": sum(r.split == "train" for r in records) >= 8000
                  and sum(r.split == "dev" for r in records) >= 512 and cfg.epochs >= 3}
    if done:
        best = {r["head"]: r["best_development_mae"] for r in receipts}
        report["candidate_improvement_over_pooled"] = best["pooled_temporal"] - best["late_temporal"]
        report["observed_threshold_met"] = report["candidate_improvement_over_pooled"] >= .02
        if report["eligible_for_outcome_gate"]:
            report["gate_passed"] = report["observed_threshold_met"]
        report["interpretation"] = "Development screening only; a small local smoke does not establish a learning gain. ABC audit is a separate condition."
    atomic_json(output / "comparison.json", report)
    return report
