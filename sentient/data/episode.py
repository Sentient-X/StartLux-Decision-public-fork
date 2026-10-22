"""One robot episode as every source adapter describes it: where its frames are and which labels it really has.

A label that a source does not have stays None; question generators only ask what the labels support. Each label
carries where it came from, because that sets how sharp its training target is (questions.py).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

Provenance = Literal["human", "operator", "program", "model", "synthetic"]
Split = Literal["train", "dev", "test"]


@dataclass(frozen=True)
class Stage:
    start: float
    end: float
    text: str


@dataclass
class Episode:
    source: str                                   # e.g. "fineart"
    episode_id: str                               # unique within the source
    family: str                                   # split unit: episodes sharing a family never cross splits
    task: str                                     # the instruction, as the source states it
    duration: float                               # seconds
    cameras: list[str]                            # camera names, as the source names them
    frame: Callable[[str, float], "object"]       # (camera, seconds) -> PIL image
    robot: Optional[str] = None
    # Outcome of the whole episode.
    success: Optional[bool] = None
    success_from: Optional[Provenance] = None
    # Subtask segmentation, in order, covering the useful part of the episode (SARM-style stages).
    stages: Optional[list[Stage]] = None
    stages_from: Optional[Provenance] = None
    # Dense task progress in [0, 1] (may fall: setbacks), and how uncertain one reading of it is.
    progress: Optional[Callable[[float], float]] = None
    progress_from: Optional[Provenance] = None
    progress_sigma: float = 0.1
    # Failure: when it first becomes visible, and what kind of failure it is.
    failure_onset: Optional[float] = None
    failure_type: Optional[str] = None
    failure_types: Optional[list[str]] = None     # the source's vocabulary of failure kinds
    failure_from: Optional[Provenance] = None
    # Human takeovers: (start, end) seconds.
    interventions: Optional[list[tuple[float, float]]] = None
    # A graded score of the whole episode: (value, labels lowest first), e.g. (3, ["1", "2", "3", "4", "5"]).
    episode_score: Optional[tuple[int, list[str]]] = None
    episode_score_from: Optional[Provenance] = None
    designated_test: bool = False                 # the source publishes this episode as test data: never train
    extra: dict = field(default_factory=dict)

    def split(self, test: float = 0.05, dev: float = 0.03) -> Split:
        """train/dev/test by a hash of the family, so whole families stay on one side; designated test stays test."""
        if self.designated_test:
            return "test"
        h = int(hashlib.sha256(f"sentient.split/v1:{self.source}:{self.family}".encode()).hexdigest()[:12], 16)
        u = h / float(1 << 48)
        return "test" if u < test else "dev" if u < test + dev else "train"


@dataclass
class Preference:
    """Two rollouts of one task, compared by people: votes for A, for B and for a tie."""
    source: str
    pair_id: str
    family: str
    task: str
    a: Episode
    b: Episode
    votes: tuple[float, float, float]
    designated_test: bool = False

    def split(self, test: float = 0.05, dev: float = 0.03) -> Split:
        return Episode.split(self, test, dev)  # type: ignore[arg-type]
