# Sentient Decision

**One calibrated model for every question about a robot episode.**

Point it at frames from any robot, state the task, and ask what you need to know:

- How far along is it?
- Which step is under way?
- Did that last second help or hurt?
- Will this attempt work out?
- Did it?
- What went wrong, and when?
- Should a person take over now?
- Which of two rollouts is better?

Every answer is a probability over the options you gave it, and the probabilities mean what they say. A reward for
RL, a value for planning, an advantage for filtering data, a success verdict for evaluation, a failure alarm for
supervision: these are all one model asked different questions.

It is a fine-tune of StartLux-Decision built with that model's own published procedure (technical report, arXiv
2610.17309). The new part is the data: robot episodes turned into many kinds of typed questions, in many formats,
with targets exactly as sharp as the labels behind them.

## What "foundational" has to mean

A model earns the word only if all of these hold, and each is measured, not asserted:

| Property | What it means | How it is measured |
|---|---|---|
| Many tasks | Value, progress, stages, advantage, outcome, failure, intervention and preference, from one set of weights | Per-question-kind scores on held-out episodes (`sentient.bench`); every kind must beat the always-answer-the-class-prior baseline |
| Many robots and scenes | Arms, bimanual rigs and mobile manipulators, labs, homes and arm farms | Held-out labs, robots and tasks that never train |
| Any phrasing | Not our wording or our JSON: thousands of state layouts, key names, question wordings, option names and orders | Every record draws its format at random; outside benchmarks use their owners' own prompts |
| Calibrated | When it says 80%, it is right 80% of the time, on every kind of question and on unseen domains | Expected calibration error and log loss per kind and per source, after a temperature refit on dev only |
| Better than the specialists | Each specialist model on its own benchmark, under its own protocol | The benchmark table below, including the results where we lose |
| Honest | Nothing from any benchmark or test split trains; families of related episodes never straddle splits | Leak checks in `sentient.data.mix` stop a build that would train on a test family |

## How one model does so many things

Every task is a typed question:

- **choice:** pick from named options;
- **noul:** yes or no;
- **score:** a level from 1 to n.

The model answers in one forward pass by reading the probability of each option's letter. The same weights answer
"which of 12 subtasks is this?", "rate completion from 0 to 100% in 5 steps" and "is a failure visible?". A new task
needs no new head, only new questions.

One FineART window of 4 frames, for example, can become all of these, each with its own target:

| Question asked | Target, from the human labels |
|---|---|
| Which step is under way? | the human subtask segment at the last frame (soft within 0.3 s of a boundary) |
| How complete is the task, in 5 levels? | progress through the human segments, spread by the label's noise |
| Did the robot move the task forward over the clip? | the sign of the progress change (never "backward" for a demonstration) |
| Is the task finished yet? | false until the last segment ends |
| Is the robot working on *this* stated task? | yes for its own task, no for another task from the same robot and scene |

The same episode gives different questions in different windows, and every question is asked only where the
source's labels can answer it. Data is used in the form it really has.

| Technical report | Here |
|---|---|
| Typed decisions with soft targets, several questions per state | `sentient/data/questions.py`: 18 question kinds, each asked only from labels that support it |
| Hash-permuted options; broad formats | `sentient/data/formats.py`: state as text or JSON in 7 layouts and 12 caption styles; wording banks; random key names, option names, order and descriptions; percent scales of 3 to 10 levels |
| Part of the earlier mixture kept | `sentient/data/retention.py`: 20% general typed decisions (SNLI, BoolQ, CommonsenseQA, ARC, HellaSwag, OpenBookQA, WinoGrande), so the model stays a general decision model |
| Soft cross-entropy on the letter read-out, LoRA adapters merged | `v1/finetune/finetune_lora.py`, unchanged except that records may carry images, read exactly as served (`check_image_readout.py`: difference 0.0 against the v1.1 engine) |
| Seed-averaged weights | `sentient/average.py` |
| Temperature per question type, refitted on held-out data | `v1/finetune/calibrate.py` |
| Reasoning supervision and RLCD² | later, where reasoning helps on hard cases |

## The questions

| Family | Kinds | What it teaches | Needs |
|---|---|---|---|
| Value | progress, remaining, time_to_go, will_succeed | how far along, how much is left, seconds until done, whether it will work out | dense progress, or success on episodes that stop when done |
| Stages (SARM-style) | stage, stage_progress | which subtask, and how far into it | stage boundaries |
| Advantage | change, compare | did this window move the task forward; which frame is further along | dense progress |
| Outcome | succeeded, done_now, end_progress, episode_score | the verdict at the end; whether it is finished yet; partial credit; graded quality | success, stage ends, end-of-episode scores |
| Grounding | instruction_match | does the footage show the stated task, against hard negatives from the same robot and scene | task text |
| Failure | failed, failure_type, failure_reason | when it went wrong, the kind of failure, why | exhaustive human failure marks |
| Intervention | takeover | a person is about to take over | intervention times |
| Preference | preference | which of two rollouts did the task better | human comparisons |

## The data

| Source | Robots and scenes | Labels (who made them) | Questions | Status |
|---|---|---|---|---|
| [FineART](https://huggingface.co/datasets/lerobot/FineART) | Trossen ALOHA-2-style bimanual, many tasks | human subtask segments | stages, progress, advantage, done_now | training (14k records); six tasks held out whole |
| [ViFailback](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset) + [FailTime](https://huggingface.co/datasets/ChangUoA/FailTime) | ALOHA, AgileX Piper, SO-101 | human failure onsets, types and reasons | failure, outcome, time_to_go | training (6k) |
| [RoboArena](https://huggingface.co/datasets/RoboArena/DataDump_02-03-2026) | DROID Franka at 7+ universities | human success, 0–100 partial success, A/B preferences | outcome, end_progress, preference | training (9k) |
| [RoboReward](https://huggingface.co/datasets/teetone/RoboReward) | 20+ Open X-Embodiment datasets: Google robot, WidowX, Franka, Jaco, UR5, xArm, PR2 and more | 1–5 end-of-episode score; generated negatives | episode_score, succeeded | training (9k of 51k); its test is RoboRewardBench, never trained |
| [BodenAI RW-RL-HIL](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset) | Galaxea R1-Lite mobile bimanual, real homes | operator takeovers in 3,347 deployments | takeover | training (6k) |
| [ArmnetBench](https://huggingface.co/datasets/armnet/armnetbench_v01_robometer) | SO-101 single-arm and bimanual arm farm; 7 policies from ACT to π0.5 plus teleoperation | human successful / suboptimal / failure | outcome, episode_score, grounding | next version; RoboJev's test sessions and its two held-out tasks are test only |
| [Robometer OOD](https://huggingface.co/datasets/robometer/rbm-1m-ood) | Trossen, Franka, Koch, xArm, SO-101 in labs never trained on | human quality labels | outcome, ranking | test only |
| [RBM-1M processed](https://huggingface.co/datasets/robometer/processed_datasets) | 94 datasets, 5.9 TB: OXE, DROID, AgiBot, LIBERO, RH20T, Galaxea, humanoids, human hands | per-trajectory quality, failure sets (LIBERO-failure, FINO-Net, FailSafe, RACER, AutoEval, SOAR) | outcome, failure, progress | next: the labelled failure sets first; full scale on Modal. Robometer's OOD domains and every benchmark's episodes excluded |
| Retention | language only | public training splits | general decisions | training (18k) |

[datasets.md](datasets.md) catalogues the wider field: DROID, AgiBot, REBOOT, RoboMIND, DenseReward, VLAC-Cut and
more, with licences and contamination risks. Licences are recorded per source and settled before any commercial
release. The StartLux-Decision weights are CC BY-NC 4.0, so commercial use of a fine-tune needs StartLux's
agreement.

**How diverse version 1 is:**

- 90,607 training questions from 5 robot sources plus retention.
- Robots: bimanual, single-arm, mobile manipulator and the full OXE spread, in labs and homes.
- Labels made by people, by programs and by policies' own outcomes.
- Every record in a format drawn at random.

Diversity is grown deliberately, not by volume: see the loop below.

## The benchmarks

Each outside benchmark is reproduced under its owner's protocol, and none of its episodes ever trains.

| Benchmark | Owner's question | Metric | Bar to beat | Base 4B | Built |
|---|---|---|---|---|---|
| **RoboRewardBench** (2,831 human-verified episodes) | 1–5 end-of-episode score | MAE ×100, mean over datasets (lower is better) | FLORA-Rewarder **40.0**; GPT-6 Astra 55.4; RoboReward-8B 66.5 | 71.0 | `sentient.roborewardbench` |
| **RoboJev SO-101** (ArmnetBench, 268 rollouts) | success, 10 frames + final views | AUROC / balanced accuracy | RoboJev-27B **0.993 / 0.952** | pending | `sentient.robojevbench` |
| **RoboJev unseen labs** (RBM-1M OOD, 459) | success | AUROC / balanced accuracy | JEV-27B-VL **0.900**; Sol **0.791** balanced accuracy | pending | `sentient.robojevbench` |
| **Robometer OOD ranking** (571) | rank failure < suboptimal < success within a task | Kendall τa ×100 | GPT-6 Astra **82.3**; FLORA 72.0; RynnValue-8B 70.4; Robometer 65.5 | pending | `sentient.robometerbench` |
| **Sentient bench** (held-out episodes of every training source) | all 18 kinds | accuracy, log loss, Brier, ECE per kind and source | beat the class prior on every kind; beat Jev 1.14 and d1-ultra | pilot: 0.70 accuracy, 0.74 log loss, 0.02 ECE (Jev 1.14: 0.59 / 1.04) | `sentient.bench` |

Next to add:

- RoboRewardBench pairwise (LLM-as-a-Verifier, 87.4%).
- ViFailback-Bench Lite (93.7).
- RoboFAC failure detection.
- RoboPulse progress pairs.
- Suboptimal-Value-Bench.

FLORA's internal DenseRewardBench and RubricBench are unreleased.

## Where it is weak, and what that changes

New question kinds and sources are added because a measured weakness asks for them, and kept only if held-out
numbers move. On the pilot's held-out test set, log loss against the class prior (lower is better):

| Kind | Pilot | Class prior | Reading |
|---|---|---|---|
| stage | 1.23 | 4.66 | learned |
| progress | 1.06 | 1.92 | learned |
| preference | 0.64 | 1.03 | learned |
| change | 0.56 | 0.66 | some |
| failed | 0.63 | 0.65 | barely |
| will_succeed | 0.37 | 0.39 | barely |
| succeeded | 0.39 | 0.37 | **not learned** |
| end_progress | 1.40 | 1.33 | **worse than the prior** |
| failure_type | 1.83 | 2.01 | weak: 33% accuracy |

The model learned to read *where in a task* a robot is. It did not learn to judge *whether the task is done*. That
is the core of every outcome benchmark, and why the pilot did not improve on RoboRewardBench (77.6 against the
base's 71.0).

The response follows the gap:

1. **More outcome data on more robots.** RoboReward's 1–5 scores (in v1), ArmnetBench and the labelled failure sets
   of RBM-1M.
2. **New outcome questions from data we already have:**
   - `done_now` turns every stage-labelled and every stop-when-done episode into "finished yet?" questions at frames
     before the end.
   - `instruction_match` teaches whether the footage shows the stated task, using other tasks from the same robot
     and scene as negatives. Success is judged against an instruction, so grounding the instruction comes first.
3. **Failure localisation before failure naming**, since naming the failure type is weak while spotting a failure
   nearly matches the prior.
4. **Whole-episode windows** for sources scored as a whole, as outcome benchmarks read them.

## Versions

Each version trains from the base model, so versions are comparable. A version trains only when its data has grown
1.5×, and is then calibrated, benched and reported to Waddle (Console /training) with no one in the loop
(`sentient.cycle`).

- The 4B model on the DGX Spark establishes that the approach works.
- Scale (9B or 27B base, RBM-1M at full size, seed averaging, Stage 3) moves to Modal once it does.

## Run

```bash
python -m sentient.data.build --source fineart --raw $RAW --out $BUILT/fineart          # per source
python -m sentient.data.retention --out $BUILT/retention
python -m sentient.data.mix --built $BUILT --out $MIX --source fineart --source failtime --source roboarena:15000
python v1/finetune/finetune_lora.py --model $BASE --train $MIX/train.jsonl --dev $MIX/dev.jsonl --out runs/s0 --seed 0
python -m sentient.average --out runs/avg runs/s0/merged runs/s1/merged runs/s2/merged
python v1/finetune/calibrate.py --model runs/avg --dev $MIX/dev.jsonl
python -m sentient.bench predict --model runs/avg --data $MIX/test.jsonl --out preds/ours.jsonl
python -m sentient.bench score preds/*.jsonl --data $MIX/test.jsonl --train $MIX/train.jsonl

# outside benchmarks, each under its owner's protocol
python -m sentient.roborewardbench build --raw $RAW --out $WORK/rrb
python -m sentient.robojevbench build --raw $RAW --out $WORK/robojev
python -m sentient.robometerbench build --raw $RAW --out $WORK/robometer

# the whole loop, as data arrives
python -m sentient.cycle --base $BASE --raw $RAW --work $WORK --runs $RUNS \
  --source fineart:14000 --source failtime --source roboarena:9000 --source roboreward:9000 --source bodenai:6000 \
  --source armnetbench --test-source robometer_ood \
  --benchmark roborewardbench=$WORK/rrb/test.jsonl --benchmark robojevbench=$WORK/robojev/test.jsonl \
  --benchmark robometerbench=$WORK/robometer/test.jsonl
```
