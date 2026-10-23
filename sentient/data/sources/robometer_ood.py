"""robometer/rbm-1m-ood (Apache-2.0): Robometer's out-of-distribution evaluation set, 571 rollouts on six robots
(Trossen, MIT Franka, USC Franka, xArm, Koch, SO-101) over 41 tasks, each labelled failure, suboptimal or successful.

Test only: it is published as an evaluation split. Questions: the quality as a three-level episode score, and the
outcome where it is unambiguous (successful vs failure; a suboptimal rollout is not asked whether it succeeded).
Clips are short (about 3 s at 10 fps, subsampled).
"""
from __future__ import annotations

from pathlib import Path

from ..episode import Episode
from ..video import VideoFrames

LEVELS = ["failure", "suboptimal", "successful"]


def items(raw: Path, shard: int, shards: int):
    import pyarrow.parquet as pq

    raw = Path(raw) / "robometer-ood"
    meta = pq.read_table(raw / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    labels = pq.read_table(raw / "data/chunk-000/file-000.parquet",
                           columns=["episode_index", "quality_label", "data_source"]).to_pandas()
    labels = labels.drop_duplicates("episode_index").set_index("episode_index")
    for m in meta:
        ep = int(m["episode_index"])
        if ep % shards != shard or ep not in labels.index:
            continue
        quality = labels.loc[ep, "quality_label"]
        if quality not in LEVELS:
            continue
        path = raw / f"videos/video/chunk-{m['videos/video/chunk_index']:03d}/file-{m['videos/video/file_index']:03d}.mp4"
        if not path.exists():
            continue
        off = float(m["videos/video/from_timestamp"])
        duration = float(m["videos/video/to_timestamp"]) - off
        yield Episode(source="robometer_ood", episode_id=str(ep), family=f"robometer_ood:{ep}", task=m["tasks"][0],
                      duration=duration, cameras=["camera"],
                      frame=lambda c, t, p=path, o=off: VideoFrames(p, o).frames([t])[t],
                      frames=lambda c, ts, p=path, o=off: VideoFrames(p, o).frames(ts),
                      robot=str(labels.loc[ep, "data_source"]).replace("_", " "),
                      success=None if quality == "suboptimal" else quality == "successful", success_from="human",
                      ends_when_done=False, episode_score=(LEVELS.index(quality), LEVELS),
                      episode_score_from="human", designated_test=True)
