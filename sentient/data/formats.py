"""Many ways to write the same record, so the model learns to answer requests, not our few templates.

Everything that is a choice of wording or layout lives here: how a state lays out the task and the frames, how a
question is phrased, what its options are called and in which order they come, how a score scale is drawn. Each is
picked with a seeded random generator, so a build is reproducible.
"""
from __future__ import annotations

import random
import string

# ---------------------------------------------------------------- state layout

TASK_KEYS = ["task", "instruction", "goal", "objective", "command", "assignment", "request", "job", "mission",
             "task_description", "language_instruction", "prompt", "what_the_robot_was_asked"]
FRAMES_KEYS = ["frames", "images", "observations", "camera_frames", "views", "snapshots", "video_frames", "history"]
ROBOT_KEYS = ["robot", "embodiment", "platform", "arm", "hardware"]
TASK_LINES = ["Task: {task}", "Instruction: {task}", "Goal: {task}", "The robot was asked to: {task}",
              "Objective — {task}", "Assigned task: \"{task}\"", "Command given to the robot: {task}",
              "{task}", "The operator instructed: {task}", "Task description: {task}", "TASK: {task}",
              "The job is to {task_lc}"]
CAMERA_ALIASES = {
    "wrist": ["wrist camera", "wrist cam", "gripper camera", "hand camera", "in-hand view", "wrist view"],
    "head": ["head camera", "head cam", "ego view", "first-person view", "head-mounted camera"],
    "front": ["front camera", "front view", "external camera", "third-person view", "scene camera"],
    "top": ["overhead camera", "top view", "top-down camera", "bird's-eye camera", "ceiling camera"],
    "side": ["side camera", "side view", "lateral camera"],
}


def camera_name(raw: str, rng: random.Random) -> str:
    low = raw.lower()
    for key, names in CAMERA_ALIASES.items():
        if key in low or (key == "front" and ("exterior" in low or "low" in low or "high" in low)):
            return rng.choice(names + [raw])
    return raw


def frame_caption(i: int, n: int, t: float, t_last: float, cam: str, style: int) -> str:
    rel = t - t_last
    return [
        f"t={t:.1f}s", f"t = {t:.2f} s", f"{rel:+.1f}s", f"frame {i + 1} of {n}", f"Frame {i + 1}",
        f"{cam}, {rel:+.1f} s", f"{cam} @ {t:.1f}s", f"[{i + 1}/{n}] {cam}",
        ("now" if i == n - 1 else f"{-rel:.1f}s ago"), ("latest" if i == n - 1 else f"earlier ({rel:.1f}s)"),
        f"step {i}", f"{cam} t{i}",
    ][style]


def state_and_images(task: str, frames: list[tuple[str, float, str]], robot: str | None, rng: random.Random):
    """frames: [(camera, t, image path)], chronological. -> (state as str or dict, image paths in mark order)."""
    t_last = max(t for _, t, _ in frames)
    cams = {c: camera_name(c, rng) for c in {c for c, _, _ in frames}}
    style = rng.randrange(12)
    n = len(frames)
    caps = [frame_caption(i, n, t, t_last, cams[c], style) for i, (c, t, _) in enumerate(frames)]
    images = [p for _, _, p in frames]
    task_lc = task[:1].lower() + task[1:] if task else task
    layout = rng.randrange(7)
    if layout <= 2:                                     # text with inline marks
        sep = rng.choice(["\n", "\n", "; ", " | "])
        lines = [rng.choice(TASK_LINES).format(task=task, task_lc=task_lc)]
        if robot and rng.random() < 0.4:
            lines.append(rng.choice(["Robot: {r}", "Embodiment: {r}", "Platform: {r}"]).format(r=robot))
        if layout == 1:
            lines.append(rng.choice(["Frames, oldest first:", "Camera frames in time order:", "What the cameras saw:",
                                     "Observations:", "Recent frames (earliest to latest):"]))
        captioned = rng.random() < 0.85
        lines.append(sep.join(f"{c}: <image>" if captioned else "<image>" for c in caps))
        return "\n".join(lines), images
    if layout == 3:                                     # images first (no marks), then the text
        head = rng.choice(["The images are frames from the robot's cameras, oldest first.",
                           "Images above: camera frames in chronological order.",
                           f"{n} frame{'s' if n > 1 else ''} from the episode, in order.", ""])
        return (head + "\n" if head else "") + rng.choice(TASK_LINES).format(task=task, task_lc=task_lc), images
    # JSON states
    tk, fk = rng.choice(TASK_KEYS), rng.choice(FRAMES_KEYS)
    state: dict[str, object] = {tk: task}
    if robot and rng.random() < 0.4:
        state[rng.choice(ROBOT_KEYS)] = robot
    if layout == 4:
        state[fk] = [{"label": c, "image": "<image>"} for c in caps]
    elif layout == 5:
        state[fk] = {c: "<image>" for c in caps} if len(set(caps)) == len(caps) else ["<image>"] * n
    else:
        state[fk] = ["<image>"] * n
        state[rng.choice(["frame_times_s", "timestamps", "t"])] = [round(t, 2) for _, t, _ in frames]
        if len(cams) > 1:
            state["cameras"] = [cams[c] for c, _, _ in frames]
    if rng.random() < 0.15:
        state[rng.choice(["note", "context", "info"])] = rng.choice(
            ["Frames are ordered from earliest to latest.", "Judge only what the frames show.",
             "Timestamps are seconds from the start of the episode.", "The last frame is the current moment."])
    if rng.random() < 0.5:
        keys = list(state)
        rng.shuffle(keys)
        state = {k: state[k] for k in keys}
    return state, images


# ---------------------------------------------------------------- question wording

NOW = ["in the latest frame", "in the last image", "now", "by the final frame", "at the end of the clip",
       "at this point", "at the most recent frame", "as of the last frame", "currently", "at the current moment"]
SUFFIX = ["", "", "", " Judge only from what is visible.", " Base the answer on the frames.",
          " Consider the stated task.", " Use the options given.", " Be careful: motion alone is not progress.",
          " Look at the objects, not just the arm.", " If unsure, pick the most likely option."]


def phrase(cores: list[str], rng: random.Random, **kw) -> str:
    text = rng.choice(cores).format(now=rng.choice(NOW), **kw)
    text = text[:1].upper() + text[1:]
    return text + rng.choice(SUFFIX)


def key_name(kind: str, rng: random.Random) -> str:
    base = rng.choice(KEYS.get(kind, [kind]))
    return base if rng.random() < 0.8 else base + "_" + rng.choice(["q", "1", "now", "est", "eval"])


KEYS = {
    "progress": ["progress", "completion", "how_far", "task_progress", "percent_done", "done_fraction", "stage_of_task"],
    "remaining": ["remaining", "left_to_do", "to_go", "work_left", "remaining_work"],
    "time_to_go": ["time_to_go", "eta", "time_left", "seconds_left", "remaining_time"],
    "stage": ["stage", "phase", "subtask", "current_step", "step", "current_phase"],
    "stage_progress": ["stage_progress", "phase_progress", "step_completion", "subtask_progress"],
    "will_succeed": ["will_succeed", "success_likely", "outcome", "will_finish", "on_track", "success"],
    "succeeded": ["succeeded", "done", "task_complete", "completed", "success", "finished"],
    "change": ["change", "delta", "progress_change", "trend", "direction", "advantage"],
    "compare": ["more_progress", "which_further", "compare", "further_along", "ahead"],
    "failed": ["failed", "failure", "went_wrong", "error_seen", "has_failed", "mistake"],
    "failure_type": ["failure_type", "what_went_wrong", "error_kind", "failure_mode", "cause"],
    "takeover": ["needs_help", "takeover_soon", "intervention", "human_needed", "will_need_help"],
    "episode_score": ["score", "rating", "quality", "grade", "episode_score"],
    "preference": ["preference", "better_rollout", "which_better", "winner", "compare_rollouts"],
}


def ordered(options: dict, rng: random.Random, keep_order: bool = False) -> dict:
    """Options in a random order (choice questions only; score levels keep their order)."""
    if keep_order:
        return options
    keys = list(options)
    rng.shuffle(keys)
    return {k: options[k] for k in keys}


def maybe_describe(options: dict[str, str], rng: random.Random, p: float = 0.6) -> dict:
    """{id: description} with descriptions kept (p) or dropped to null."""
    keep = rng.random() < p
    return {k: (v if keep else None) for k, v in options.items()}


PERCENT_WORDS = {3: ["just started", "about halfway", "almost or fully done"],
                 4: ["barely started", "under half done", "over half done", "nearly or fully done"],
                 5: ["not started", "early", "about halfway", "nearly done", "done"],
                 6: ["nothing done", "a little", "a third", "two thirds", "almost done", "complete"]}


def percent_scale(n: int, rng: random.Random) -> list[str]:
    """Descriptions of n equal levels of a [0, 1] quantity, lowest first, in one of several styles."""
    style = rng.randrange(4)
    if style == 0 and n in PERCENT_WORDS:
        return PERCENT_WORDS[n]
    edges = [round(100 * i / n) for i in range(n + 1)]
    if style == 1:
        return [f"{a}-{b}%" for a, b in zip(edges, edges[1:])]
    if style == 2:
        return [f"{a}% to {b}%" for a, b in zip(edges, edges[1:])]
    return [f"level {i} ({a}-{b}% complete)" for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def letters_or_names(n: int, rng: random.Random) -> list[str]:
    style = rng.randrange(3)
    if style == 0:
        return list(string.ascii_uppercase[:n])
    if style == 1:
        return [f"option_{i + 1}" for i in range(n)]
    return [str(i + 1) for i in range(n)]
