"""Paired episode-family uncertainty for the local development screen."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics

from .data import atomic_json, file_digest, read_manifest


def paired_comparison(candidate: list[dict], control: list[dict], draws: int = 10000, seed: int = 17) -> dict:
    left, right = {r["id"]: r for r in candidate}, {r["id"]: r for r in control}
    if len(left) != len(candidate) or len(right) != len(control) or not left or left.keys() != right.keys():
        raise ValueError("Predictions must have identical unique membership")
    groups = defaultdict(list)
    for key, a in left.items():
        b = right[key]
        if (a["family"], a["gold"]) != (b["family"], b["gold"]):
            raise ValueError("Paired predictions disagree on family or target")
        groups[a["family"]].append(abs(b["prediction"] - b["gold"]) - abs(a["prediction"] - a["gold"]))
    clusters = [(sum(values), len(values)) for _, values in sorted(groups.items())]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sample = rng.choices(clusters, k=len(clusters))
        estimates.append(sum(total for total, _ in sample) / sum(n for _, n in sample))
    estimates.sort()
    return {"mae_improvement": sum(s for s, _ in clusters) / sum(n for _, n in clusters),
            "ci95_episode_bootstrap": [estimates[int(.025 * draws)], estimates[min(draws - 1, int(.975 * draws))]],
            "families": len(clusters), "examples": len(left), "bootstrap_draws": draws, "seed": seed,
            "sign": "positive favors late_temporal"}


def analyze(manifest: Path, training: Path) -> dict:
    from .heads import HEAD_KINDS
    records, manifest_digest = read_manifest(manifest)
    dev = {r.id: r for r in records if r.split == "dev"}
    training_labels = [r.reward for r in records if r.split == "train"]
    constant = int(statistics.median_low(training_labels))
    report = {"manifest_digest": manifest_digest,
              "interpretation": "One-seed development screen, not test-set or FLORA superiority. Best-epoch intervals are conditional on development selection; final-epoch intervals avoid epoch selection but remain development evidence.",
              "train_label_counts": dict(Counter(training_labels)),
              "dev_label_counts": dict(Counter(r.reward for r in dev.values())),
              "training_median_baseline": {"prediction": constant,
                  "development_mae": sum(abs(r.reward - constant) for r in dev.values()) / len(dev)},
              "epochs": {}}
    for label, filename in (("best", "predictions.json"), ("final", "last_predictions.json")):
        predictions, summaries = {}, {}
        for kind in HEAD_KINDS:
            path = training / kind / filename
            rows = json.loads(path.read_text())
            if len(rows) != len(dev) or {r["id"] for r in rows} != dev.keys():
                raise ValueError("Prediction membership differs from development manifest")
            by_source = defaultdict(list)
            for row in rows:
                record = dev[row["id"]]
                if (row["gold"], row["family"]) != (record.reward, record.family):
                    raise ValueError("Prediction target/lineage differs from manifest")
                by_source[record.family.split("/")[0]].append(abs(row["prediction"] - record.reward))
            summaries[kind] = {"mae": sum(abs(r["prediction"] - r["gold"]) for r in rows) / len(rows),
                               "prediction_counts": dict(Counter(r["prediction"] for r in rows)),
                               "source_mae": {s: {"n": len(v), "mae": sum(v) / len(v)} for s, v in sorted(by_source.items())},
                               "prediction_sha256": file_digest(path)}
            predictions[kind] = rows
        report["epochs"][label] = {"heads": summaries,
            "paired": {kind: paired_comparison(predictions["late_temporal"], predictions[kind])
                       for kind in HEAD_KINDS if kind != "late_temporal"}}
    atomic_json(training / "uncertainty.json", report)
    return report
