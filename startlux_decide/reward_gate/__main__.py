"""Offline, bounded research gate; no remote compute or labeling jobs are launched."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Mint source-family-disjoint RoboReward membership")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--train-count", type=int, default=96)
    prepare.add_argument("--dev-count", type=int, default=32)
    prepare.add_argument("--seed", type=int, default=17)
    cache = sub.add_parser("cache", help="Encode pinned frozen SigLIP features once")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    cache.add_argument("--media-budget-mb", type=int, default=128)
    cache.add_argument("--feature-budget-mb", type=int, default=512)
    cache.add_argument("--local-model", action="store_true", help="Require already-cached pinned encoder")
    cache.add_argument("--stream-media", action="store_true", help="Verify media in RAM without retaining another disk copy")
    cache.add_argument("--compressed", action="store_true", help="Losslessly compress cached feature bytes")
    cache.add_argument("--reuse", type=Path, help="Reuse compatible content-addressed features from a prior cache")
    train = sub.add_parser("train", help="Train all three controls with resumable checkpoints")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--microbatch", type=int, default=8)
    train.add_argument("--effective-batch", type=int, default=64)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--max-updates", type=int, help="Persist and return after this many updates per head")
    analysis = sub.add_parser("analyze", help="Compare paired development predictions with episode bootstrap")
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--training", type=Path, required=True)
    abc = sub.add_parser("inspect-abc", help="Mine annotation hints; never invent recovery labels")
    abc.add_argument("--output", type=Path, required=True)
    abc.add_argument("--max-episodes", type=int, default=30)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare as prepare_data
        result = prepare_data(args.manifest, args.train_count, args.dev_count, args.seed)
        result = {k: v for k, v in result.items() if k != "records"} | {"records": len(result["records"])}
    elif args.command == "cache":
        from .features import FrozenSiglip, build_cache
        encoder = FrozenSiglip(args.device, args.local_model)
        result = build_cache(args.manifest, args.output, encoder, args.media_budget_mb, args.feature_budget_mb,
                             stream_media=args.stream_media, compressed=args.compressed, reuse=args.reuse)
        result = {"complete": result["complete"], "records": len(result["rows"]), "encoder_digest": result["encoder_digest"]}
    elif args.command == "train":
        from .train import TrainConfig, run
        result = run(args.manifest, args.cache, args.output,
                     TrainConfig(args.epochs, args.microbatch, args.effective_batch, seed=args.seed, device=args.device),
                     args.max_updates)
    elif args.command == "analyze":
        from .analysis import analyze
        result = analyze(args.manifest, args.training)
    else:
        from .abc import inspect_abc
        result = inspect_abc(args.output, max_episodes=args.max_episodes)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
