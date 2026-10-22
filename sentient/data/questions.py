"""Typed questions an episode's labels can answer about a window of its frames, each with a soft target.

Every generator returns None when the episode lacks the label it needs, so a source contributes exactly the questions
its labels support. Targets follow the fine-tuning format (v1/docs/finetuning.md): criteria keys for choice,
"true"/"false" for noul, "0".."n-1" for score.

The kinds, by what they teach:
  progress, remaining, time_to_go, will_succeed  value: how far along, how much is left, will it work out
  stage, stage_progress                          SARM-style: which subtask, and how far into it
  change, compare                                advantage: did the window move the task forward, which frame is ahead
  failed, failure_type, takeover                 failure onset, its kind, and an imminent human takeover
  succeeded, episode_score                       outcome of the whole episode
"""
from __future__ import annotations

import random
from typing import Callable, Optional

from . import formats as F
from .episode import Episode
from .targets import levels, three_way, tidy, yes_probability

SIGMA = {"human": 0.05, "operator": 0.05, "program": 0.08, "model": 0.15, "synthetic": 0.12}
Q = tuple[str, dict, dict]          # (kind, question spec, target)


def _noul(text: str, p_yes: float, rng: random.Random, yes: str | None = None, no: str | None = None) -> tuple[dict, dict]:
    spec: dict[str, object] = {"type": "noul", "instructions": text}
    if (yes or no) and rng.random() < 0.5:
        spec["criteria"] = {"true": yes, "false": no}
    return spec, tidy({"true": p_yes, "false": 1 - p_yes})


def _score(text: str, descriptions: list[str], dist: list[float]) -> tuple[dict, dict]:
    return ({"type": "score", "instructions": text, "criteria": descriptions},
            tidy({str(i): p for i, p in enumerate(dist)}))


def progress(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.progress is None:
        return None
    n = rng.choice([3, 4, 5, 5, 6, 8, 10])
    p = min(max(ep.progress(ts[-1]), 0.0), 1.0)
    text = F.phrase(["How far along is the task {now}?", "How much of the task has been completed {now}?",
                     "Estimate the robot's progress on the task {now}.", "Rate how complete the task is {now}.",
                     "Where is the robot in completing its task {now}?", "What share of the task is done {now}?",
                     "Judge task completion {now}.", "How close is the robot to finishing {now}?"], rng)
    return ("progress", *_score(text, F.percent_scale(n, rng), levels(p, n, SIGMA.get(ep.progress_from or '', 0.1) + ep.progress_sigma * 0.5)))


def remaining(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.progress is None:
        return None
    n = rng.choice([3, 4, 5])
    p = 1 - min(max(ep.progress(ts[-1]), 0.0), 1.0)
    text = F.phrase(["How much of the task is still left to do {now}?", "How much work remains {now}?",
                     "What fraction of the task remains {now}?"], rng)
    descs = [d.replace("complete", "remaining") for d in F.percent_scale(n, rng)]
    return ("remaining", *_score(text, descs, levels(p, n, SIGMA.get(ep.progress_from or '', 0.1))))


def time_to_go(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.success is not True:
        return None
    left = ep.duration - ts[-1]
    edges = rng.choice([[0, 5, 15, 30, 1e9], [0, 3, 10, 20, 40, 1e9], [0, 10, 30, 1e9]])
    names = []
    for a, b in zip(edges, edges[1:]):
        names.append(f"under {b:g} s" if a == 0 else f"more than {a:g} s" if b > 1e8 else f"{a:g}-{b:g} s")
    from .targets import binned
    dist = binned(left, edges, max(0.5, 0.1 * left))
    text = F.phrase(["The robot will finish this task. About how long from the last frame until it is done?",
                     "This episode ends in success. Roughly how many seconds of work are left after the latest frame?",
                     "Given that the robot completes the task, how much time remains {now}?"], rng)
    return ("time_to_go", *_score(text, names, dist))


def will_succeed(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.success is None or ts[-1] > ep.duration - 0.5:
        return None
    text = F.phrase(["Will the robot complete the task in this episode?", "Is this attempt going to succeed?",
                     "Judging from the frames so far, will the task end in success?",
                     "Is the robot on track to finish the task successfully?",
                     "Predict the outcome: will the task be completed?"], rng)
    return ("will_succeed", *_noul(text, 1.0 if ep.success else 0.0, rng, "the episode ends with the task done",
                                   "the episode ends without the task done"))


def succeeded(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.success is None or ts[-1] < ep.duration - 0.5:
        return None
    text = F.phrase(["Did the robot complete the task?", "Was the task accomplished by the end?",
                     "Is the task done in the final frame?", "Did this episode succeed?",
                     "Has the instruction been carried out successfully?"], rng)
    return ("succeeded", *_noul(text, 1.0 if ep.success else 0.0, rng, "task completed", "task not completed"))


def _stage_at(ep: Episode, t: float) -> dict[int, float]:
    """Stage index -> probability at time t; soft within 0.3 s of a boundary (human boundaries are that loose)."""
    stages = ep.stages or []
    width = 0.3 if ep.stages_from in ("human", "operator") else 0.8
    w = {}
    for i, s in enumerate(stages):
        inside = min(t - s.start, s.end - t)
        w[i] = yes_probability(inside, width / 3)
    total = sum(w.values())
    return {i: v / total for i, v in w.items() if v / total > 1e-4} if total > 0 else {}


def stage(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if not ep.stages or len(ep.stages) < 2:
        return None
    dist = _stage_at(ep, ts[-1])
    if not dist:
        return None
    texts = []
    for s in ep.stages:
        if s.text not in texts:
            texts.append(s.text)
    if len(texts) < 2 or len(texts) > 20 or any(len(t) > 150 or "\n" in t for t in texts):
        return None
    named = rng.random() < 0.5
    ids = texts if named else F.letters_or_names(len(texts), rng)
    options = {i: (None if named else t) for i, t in zip(ids, texts)}
    target: dict[str, float] = {}
    for k, p in dist.items():
        target[ids[texts.index(ep.stages[k].text)]] = target.get(ids[texts.index(ep.stages[k].text)], 0) + p
    text = F.phrase(["Which step of the task is the robot doing {now}?", "What subtask is in progress {now}?",
                     "Which phase of the task is the robot in {now}?", "Identify the current step {now}.",
                     "Which of these steps best describes what the robot is doing {now}?"], rng)
    return ("stage", {"type": "choice", "instructions": text, "criteria": F.ordered(options, rng)}, tidy(target))


def stage_progress(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if not ep.stages:
        return None
    dist = _stage_at(ep, ts[-1])
    if not dist or max(dist.values()) < 0.95:
        return None
    s = ep.stages[max(dist, key=lambda k: dist[k])]
    p = (ts[-1] - s.start) / max(s.end - s.start, 1e-3)
    n = rng.choice([3, 4, 5])
    text = rng.choice(["The robot is working on this step: \"{s}\". How far through this step is it {now}?",
                       "Current subtask: {s}. How much of this subtask is done {now}?",
                       "Considering only the step \"{s}\", how complete is it {now}?"]).format(s=s.text, now=rng.choice(F.NOW))
    return ("stage_progress", *_score(text, F.percent_scale(n, rng), levels(p, n, 0.12)))


CHANGE_NAMES = [("advance", "no_change", "setback"), ("better", "same", "worse"), ("progressed", "unchanged", "regressed"),
                ("improved", "no change", "got worse"), ("forward", "stalled", "backward"), ("up", "flat", "down"),
                ("progress", "no progress", "lost progress")]
CHANGE_DESC = ("the task is further along at the end than at the start",
               "task progress is the same at both ends of the window",
               "progress that existed at the start has been lost by the end")


def change(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.progress is None or len(ts) < 2:
        return None
    d = ep.progress(ts[-1]) - ep.progress(ts[0])
    sigma = SIGMA.get(ep.progress_from or '', 0.1) * 0.7
    up, flat, down = three_way(d, 0.04, sigma)
    if ep.monotone:                       # the reading is uncertain in size, not in sign
        up, flat, down = (up / (up + flat), flat / (up + flat), 0.0) if up + flat > 0 else (0.0, 1.0, 0.0)
    names = rng.choice(CHANGE_NAMES)
    options = F.maybe_describe(dict(zip(names, CHANGE_DESC)), rng)
    text = F.phrase(["Between the first and the last frame, how did task progress change?",
                     "Compare the earliest and latest frames: did the robot move the task forward, stay put, or lose ground?",
                     "Over this clip, did the robot make progress on the task?",
                     "From the start of the clip to its end, what happened to task progress?",
                     "Did the robot's actions in this window help, do nothing, or undo progress?"], rng)
    return ("change", {"type": "choice", "instructions": text, "criteria": F.ordered(options, rng)},
            tidy({names[0]: up, names[1]: flat, names[2]: down}))


def compare(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    """Which of two frames (named, shown in time order) is further along; the reading noise sets the softness."""
    if ep.progress is None or len(ts) < 2:
        return None
    a, b = ts[0], ts[-1]
    d = ep.progress(b) - ep.progress(a)
    names = rng.choice([("first frame", "last frame"), ("earliest frame", "latest frame"), ("start of the clip", "end of the clip")])
    sigma = SIGMA.get(ep.progress_from or '', 0.1)
    tie = rng.random() < 0.5
    flat = three_way(d, 0.03, sigma)[1] if tie else 0.0
    if ep.monotone:                       # the earlier frame is never ahead in a demonstration
        if not tie and d < 0.03:
            return None
        p_b = 1.0
    else:
        p_b = yes_probability(d, sigma)
    target = {names[1]: p_b * (1 - flat), names[0]: (1 - p_b) * (1 - flat)}
    if tie:
        target["about the same"] = flat
    options = {k: None for k in target}
    text = F.phrase(["Which shows the task further along: the {a} or the {b}?",
                     "Is the robot closer to finishing at the {a} or at the {b}?"], rng, a=names[0], b=names[1])
    return ("compare", {"type": "choice", "instructions": text, "criteria": F.ordered(options, rng)}, tidy(target))


def failed(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.failure_onset is None and ep.success is not True:
        return None
    t = ts[-1]
    p = 0.0 if ep.failure_onset is None else yes_probability(t - ep.failure_onset, 0.15 if ep.failure_from == "human" else 0.4)
    text = F.phrase(["Has something gone wrong with the task {now}?", "Has the robot made a mistake or failed {now}?",
                     "Is there a visible failure {now}?", "Has the attempt failed by the last frame?",
                     "Did an error happen at any point up to the latest frame?"], rng)
    return ("failed", *_noul(text, p, rng, "a failure is visible", "no failure so far"))


def failure_type(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.failure_onset is None or not ep.failure_type or not ep.failure_types or ts[-1] < ep.failure_onset + 0.5:
        return None
    kinds = list(dict.fromkeys(ep.failure_types))
    if ep.failure_type not in kinds or len(kinds) < 2:
        return None
    text = F.phrase(["What went wrong?", "Which kind of failure is shown?", "Classify the failure visible in the frames.",
                     "What is the cause of the failure?"], rng)
    return ("failure_type", {"type": "choice", "instructions": text,
                             "criteria": F.ordered({k: None for k in kinds}, rng)}, {ep.failure_type: 1.0})


def failure_reason(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    """Which written observation matches what went wrong, among others from the same task (human reasons)."""
    true = ep.extra.get("failure_reason")
    pool = [r for r in dict.fromkeys(ep.extra.get("reason_pool") or []) if r != true]
    if not true or ep.failure_onset is None or ts[-1] < ep.failure_onset + 0.5 or len(pool) < 2:
        return None
    others = rng.sample(pool, min(len(pool), rng.choice([2, 3, 4])))
    ids = F.letters_or_names(len(others) + 1, rng)
    texts = [true] + others
    rng.shuffle(texts)
    if any(len(t) > 400 for t in texts):
        return None
    options = dict(zip(ids, texts))
    text = F.phrase(["Which description matches what went wrong?", "What happened in this failed attempt?",
                     "Pick the account of the failure that fits the frames."], rng)
    return ("failure_reason", {"type": "choice", "instructions": text, "criteria": options},
            {ids[texts.index(true)]: 1.0})


def takeover(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.interventions is None:
        return None
    t = ts[-1]
    if any(s <= t <= e for s, e in ep.interventions):
        return None                                   # a person is already driving
    horizon = rng.choice([2.0, 3.0, 5.0])
    soon = min((s - t for s, _ in ep.interventions if s > t), default=1e9)
    p = yes_probability(horizon - soon, 0.4)
    text = rng.choice(["Will a person need to take over the robot within the next {h:g} seconds?",
                       "Is a human intervention about to be needed (within {h:g} s)?",
                       "Should a supervisor be ready to step in within {h:g} seconds?"]).format(h=horizon)
    return ("takeover", *_noul(text, p, rng))


def end_progress(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    """How much of the task was done by the end, from a person's end-of-episode partial-success score."""
    p = ep.extra.get("final_progress")
    if p is None or ts[-1] < ep.duration - 0.5:
        return None
    n = rng.choice([3, 4, 5, 6])
    text = F.phrase(["How much of the task did the robot get done by the end?",
                     "By the final frame, what share of the task has been accomplished?",
                     "Rate the partial success of this attempt at its end."], rng)
    return ("end_progress", *_score(text, F.percent_scale(n, rng),
                                    levels(min(max(p, 0.0), 1.0), n, ep.extra.get("final_progress_sigma", 0.08))))


def episode_score(ep: Episode, ts: list[float], rng: random.Random) -> Optional[Q]:
    if ep.episode_score is None or ts[-1] < ep.duration - 0.5:
        return None
    value, names = ep.episode_score
    text = F.phrase(["Rate how well the robot did the task over this episode.", "Score the episode.",
                     "How successful was this rollout overall?"], rng)
    return ("episode_score", {"type": "score", "instructions": text, "criteria": names},
            tidy({str(i): (1.0 if i == value else 0.0) for i in range(len(names))}))


GENERATORS: dict[str, Callable] = {
    "progress": progress, "remaining": remaining, "time_to_go": time_to_go, "will_succeed": will_succeed,
    "succeeded": succeeded, "stage": stage, "stage_progress": stage_progress, "change": change, "compare": compare,
    "failed": failed, "failure_type": failure_type, "failure_reason": failure_reason, "takeover": takeover, "end_progress": end_progress, "episode_score": episode_score,
}
