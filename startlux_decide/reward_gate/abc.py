"""Bounded ABC annotation inspection; mining hints never become reward labels.

Run ``python -m startlux_decide.reward_gate.abc --output runs/reward-gate/abc``.
Only existing Hugging Face authorization is used. No video is downloaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
from typing import Optional


REPO_ID = "XDOF/ABC-130k"
REVISION = "d4c79f5aecb902f3157e6139661cb1b0b2a36689"
MAX_EPISODES = 300
MAX_BYTES = 50 * 1024 * 1024
TOPIC = "/subtask-annotation"
HINT = re.compile(
    r"\b(?:correct(?:ion|ing|ed)?|mistakes?|reset(?:ting)?|retry|retrying|"
    r"retries|recover(?:y|ing|ed)?|regrasp|re-grasp|slip(?:ped|ping)?|"
    r"drop(?:ped|ping)?|fail(?:ed|ure|ing)?)\b", re.IGNORECASE
)


class AnnotationSchemaError(ValueError):
    """An annotation does not match the inspected pinned schema."""


@dataclass(frozen=True)
class Transition:
    start_ns: int
    action: str


@dataclass(frozen=True)
class Interval:
    start_ns: int
    end_ns: Optional[int]
    action: str


def normalize_action(action: str) -> str:
    """Conservative exact-repeat key; do not infer semantic equivalence."""
    return " ".join(re.sub(r"[^\w\s]", " ", action.casefold()).split())


def intervals_from_transitions(transitions: list[Transition]) -> list[Interval]:
    """Use the next transition as end; final episode bound is unavailable."""
    ordered = sorted(transitions, key=lambda row: row.start_ns)
    for i, row in enumerate(ordered):
        if row.start_ns < 0 or not row.action.strip():
            raise AnnotationSchemaError("invalid timestamp or empty action")
        if i and row.start_ns == ordered[i - 1].start_ns:
            raise AnnotationSchemaError("simultaneous transitions need manual review")
    # Consecutive identical text is one continuous interval, not a retry hint.
    merged: list[Transition] = []
    for row in ordered:
        if not merged or normalize_action(row.action) != normalize_action(merged[-1].action):
            merged.append(row)
    return [
        Interval(row.start_ns, merged[i + 1].start_ns if i + 1 < len(merged) else None, row.action)
        for i, row in enumerate(merged)
    ]


def parse_annotation(path: Path) -> tuple[list[Interval], list[dict]]:
    """Decode actual embedded protobuf; reject unknown or conflicting schemas."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    transitions: list[Transition] = []
    schemas: dict[str, dict] = {}
    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic != TOPIC:
                raise AnnotationSchemaError("unexpected topic in annotation-only file")
            if schema is None or schema.name != "Annotation" or schema.encoding != "protobuf":
                raise AnnotationSchemaError("unexpected annotation schema")
            descriptor = decoded.DESCRIPTOR
            fields = descriptor.fields_by_name
            if set(fields) != {"timestamp", "data"}:
                raise AnnotationSchemaError("expected only timestamp and data fields")
            if fields["data"].type != fields["data"].TYPE_STRING:
                raise AnnotationSchemaError("data must be a string")
            timestamp_type = fields["timestamp"].message_type
            if timestamp_type is None or timestamp_type.full_name != "google.protobuf.Timestamp":
                raise AnnotationSchemaError("timestamp must be google.protobuf.Timestamp")
            if not decoded.HasField("timestamp") or not 0 <= decoded.timestamp.nanos < 1_000_000_000:
                raise AnnotationSchemaError("invalid protobuf timestamp")
            start_ns = decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos
            if start_ns != message.log_time:
                raise AnnotationSchemaError("payload and envelope timestamps disagree")
            transitions.append(Transition(start_ns, decoded.data))
            schemas[schema.name] = {
                "name": schema.name, "encoding": schema.encoding,
                "topic": channel.topic, "fields": sorted(fields),
                "descriptor_sha256": hashlib.sha256(schema.data).hexdigest(),
            }
    if not transitions:
        raise AnnotationSchemaError("no annotation transitions found")
    return intervals_from_transitions(transitions), list(schemas.values())


def mine_candidates(intervals: list[Interval], source_path: str) -> tuple[list[dict], list[dict]]:
    """Return review candidates and ordinary controls, never supervision labels."""
    if not _is_train_annotation(source_path):
        raise ValueError("only train annotation paths are accepted")
    counts = Counter(normalize_action(row.action) for row in intervals)
    first_seen: dict[str, int] = {}
    candidates: list[dict] = []
    controls: list[dict] = []
    for index, interval in enumerate(intervals):
        key = normalize_action(interval.action)
        words = sorted({m.group(0).casefold() for m in HINT.finditer(interval.action)})
        repeated = counts[key] > 1
        record = {
            "source_path": source_path,
            "episode_id": PurePosixPath(source_path).parent.name,
            "task_from_path": PurePosixPath(source_path).parts[2],
            "interval_index": index,
            **asdict(interval),
            "time_basis": "absolute_unix_nanoseconds",
            "end_known": interval.end_ns is not None,
            "verification_status": "unverified",
            "training_eligible": False,
            "human_label": None,
            "reasons": (["repeated_action"] if repeated else []) + (["correction_keyword"] if words else []),
            "matched_keywords": words,
            "previous_same_action_index": first_seen.get(key),
        }
        if repeated or words:
            candidates.append(record)
        elif interval.end_ns is not None:
            controls.append(record)
        first_seen[key] = index
    return candidates, controls


def _is_train_annotation(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return (
        len(parts) == 5 and parts[:2] == ("data", "train")
        and parts[3].startswith("episode_") and parts[4] == "annotation.mcap"
        and all(part not in {".", ".."} for part in parts)
    )


def _error_evidence(exc: Exception, operation: str) -> dict:
    """No exception message, URL, request headers, or token enters a report."""
    response = getattr(exc, "response", None)
    return {"operation": operation, "exception_type": type(exc).__name__,
            "http_status": getattr(response, "status_code", None)}


def inspect_abc(
    output_dir: Path | str = "runs/reward-gate/abc", *, max_episodes: int = 30,
    max_bytes: int = MAX_BYTES, seed: int = 1729, max_tree_entries: int = 30_000,
) -> dict:
    """Inspect bounded train annotations and persist report.json.

    Downloads are preflighted against sizes from the pinned HF tree and metadata;
    no snapshot, video, held-out split, login, or terms-acceptance operation exists.
    The annotation artifact budget counts cached files too, making reruns bounded.
    """
    if not 1 <= max_episodes <= MAX_EPISODES:
        raise ValueError("max_episodes must be between 1 and 300")
    if not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("max_bytes must be between 1 and 50 MiB")
    if not 1 <= max_tree_entries <= 100_000:
        raise ValueError("max_tree_entries must be between 1 and 100000")
    from huggingface_hub import HfApi, hf_hub_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": REPO_ID, "revision": REVISION, "split": "train", "seed": seed,
        "status": "complete", "max_episodes": max_episodes, "max_bytes": max_bytes,
        "annotation_bytes": 0, "episodes_attempted": 0, "episodes_parsed": 0,
        "tree_entries_seen": 0, "tasks_visited": [], "episodes": [],
        "candidates": [], "ordinary_controls": [], "errors": [],
        "limitations": [
            "Annotation-only mining: no video or physical mistake/recovery was verified.",
            "Repeated actions can be ordinary repetition; keywords can describe intended actions.",
            "Task names come from paths; episode instructions and final time bounds were not downloaded.",
            "Task-shuffled bounded metadata traversal is a convenience sample, not a prevalence estimate.",
        ],
        "retry_command": (
            "python -m startlux_decide.reward_gate.abc --output " + str(output_dir)
            + " --max-episodes " + str(max_episodes) + " --max-bytes " + str(max_bytes)
            + " --seed " + str(seed)
        ),
    }
    api = HfApi()  # HF's default resolution reuses existing environment/saved auth.
    artifact_bytes = 0
    controls: list[dict] = []
    task_generators = []
    try:
        tasks = [item.path for item in api.list_repo_tree(
            REPO_ID, path_in_repo="data/train", repo_type="dataset", revision=REVISION,
        ) if item.path.count("/") == 2 and not hasattr(item, "size")]
        random.Random(seed).shuffle(tasks)
        for task in tasks:
            task_generators.append((task, iter(api.list_repo_tree(
                REPO_ID, path_in_repo=task, repo_type="dataset", revision=REVISION, recursive=True,
            ))))
        # One annotation per task per pass, so a single dense task does not dominate.
        while task_generators and report["episodes_attempted"] < max_episodes:
            remaining = []
            for task, items in task_generators:
                if report["episodes_attempted"] >= max_episodes or report["tree_entries_seen"] >= max_tree_entries:
                    break
                if task not in report["tasks_visited"]:
                    report["tasks_visited"].append(task)
                selected = None
                # Bound metadata-only work on tasks lacking annotations.
                for _ in range(min(2000, max_tree_entries - report["tree_entries_seen"])):
                    item = next(items, None)
                    if item is None:
                        break
                    report["tree_entries_seen"] += 1
                    if _is_train_annotation(item.path):
                        selected = item
                        break
                if selected is None:
                    continue
                remaining.append((task, items))
                size = selected.size
                if not isinstance(size, int) or size <= 0 or artifact_bytes + size > max_bytes:
                    report["limitations"].append("An annotation was skipped because its advertised size exceeded the artifact budget.")
                    continue
                report["episodes_attempted"] += 1
                artifact_bytes += size
                report["annotation_bytes"] = artifact_bytes
                episode = {"source_path": selected.path, "advertised_bytes": size,
                           "episode_id": PurePosixPath(selected.path).parent.name,
                           "episode_video_path": str(PurePosixPath(selected.path).parent / "episode.mcap"),
                           "episode_video_advertised_bytes": None}
                report["episodes"].append(episode)
                try:
                    local = Path(hf_hub_download(
                        REPO_ID, selected.path, repo_type="dataset", revision=REVISION,
                        local_dir=output_dir / "source",
                    ))
                    if local.stat().st_size != size:
                        raise AnnotationSchemaError("download size disagrees with pinned tree")
                    episode["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
                    intervals, schemas = parse_annotation(local)
                    candidates, ordinary = mine_candidates(intervals, selected.path)
                    report["candidates"].extend(candidates)
                    controls.extend(ordinary)
                    episode.update({"interval_count": len(intervals), "candidate_count": len(candidates),
                                    "ordinary_count": len(ordinary), "schemas": schemas})
                    report["episodes_parsed"] += 1
                except Exception as exc:
                    evidence = _error_evidence(exc, "download_or_parse_annotation")
                    evidence["source_path"] = selected.path
                    report["errors"].append(evidence)
                    if evidence["http_status"] in {401, 403}:
                        report["status"] = "access_denied"
                        raise PermissionError("existing dataset access is insufficient") from None
            task_generators = remaining
            if report["tree_entries_seen"] >= max_tree_entries:
                report["limitations"].append("Metadata traversal limit reached before exhausting the dataset.")
                break
        # File sizes only; no video content is requested.
        if report["episodes"]:
            try:
                paths = [ep["episode_video_path"] for ep in report["episodes"]]
                sizes = {item.path: getattr(item, "size", None) for item in api.get_paths_info(
                    REPO_ID, paths, repo_type="dataset", revision=REVISION,
                )}
                for episode in report["episodes"]:
                    episode["episode_video_advertised_bytes"] = sizes.get(episode["episode_video_path"])
            except Exception as exc:
                report["errors"].append(_error_evidence(exc, "video_sizes_metadata_only"))
    except Exception as exc:
        report["errors"].append(_error_evidence(exc, "train_inventory_or_access"))
        if report["status"] != "access_denied":
            status = getattr(getattr(exc, "response", None), "status_code", None)
            report["status"] = "access_denied" if status in {401, 403} else "incomplete"
    if report["episodes_parsed"] < max_episodes and report["status"] == "complete":
        report["status"] = "partial"
    if report["errors"] and report["status"] == "complete":
        report["status"] = "partial"
    if report["status"] == "access_denied":
        report["limitations"].append("Existing account access denied; no terms accepted and no access bypass attempted.")
    random.Random(seed).shuffle(controls)
    report["ordinary_controls"] = controls[:max_episodes * 2]
    report["counts"] = {
        "candidate_intervals": len(report["candidates"]),
        "candidate_episodes": len({row["episode_id"] for row in report["candidates"]}),
        "ordinary_intervals_available": len(controls),
        "ordinary_controls_sampled": len(report["ordinary_controls"]),
        "verified_mistakes": 0, "verified_recoveries": 0,
        "keyword_intervals": sum("correction_keyword" in row["reasons"] for row in report["candidates"]),
        "repeated_action_intervals": sum("repeated_action" in row["reasons"] for row in report["candidates"]),
    }
    parsed = report["episodes_parsed"]
    report["candidate_episode_yield"] = report["counts"]["candidate_episodes"] / parsed if parsed else None
    report_path = output_dir / "report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/reward-gate/abc"))
    parser.add_argument("--max-episodes", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-tree-entries", type=int, default=30_000)
    args = parser.parse_args(argv)
    report = inspect_abc(args.output, max_episodes=args.max_episodes, max_bytes=args.max_bytes,
                         seed=args.seed, max_tree_entries=args.max_tree_entries)
    print(json.dumps({"report": str(args.output / "report.json"), "status": report["status"],
                      "episodes_parsed": report["episodes_parsed"], "annotation_bytes": report["annotation_bytes"],
                      "counts": report["counts"]}, indent=2))
    return 0 if report["status"] in {"complete", "partial"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
