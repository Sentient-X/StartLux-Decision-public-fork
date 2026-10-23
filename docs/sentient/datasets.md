# Public datasets for Sentient Decision (reward / value / advantage VLM)

Researched 2026-10-22 from Hugging Face cards and the HF API, arXiv pages and GitHub repos. Nothing was downloaded.
Most cards were read through a summarising fetch tool. Before ingesting a source, re-read its licence line yourself.
Where a fact is unverified, the table says so.

**Licence key:**
- **OK:** a permissive licence that allows commercial use.
- **NC:** non-commercial. **We cannot use these for a commercial model.**
- **?:** unknown or unstated. Treat it as all rights reserved until we have written terms.
- **G:** gated (click-through or a contact form).

**Question key:**
- **prog:** progress
- **stage:** stage or subtask
- **succ:** success
- **fail:** failure onset or type
- **pref:** pairwise preference
- **val:** value / time-to-go
- **adv:** advantage, meaning the progress change between two frames, or intervention as a negative-advantage signal

## Main table

### 1. Dense progress / reward

| Dataset | Date | Size / partial | Real? | Embodiments | Labels (provenance) | Licence | Questions | Held-out |
|---|---|---|---|---|---|---|---|---|
| [Robometer RBM-1M](https://robometer.github.io/) | 2026-03 (2603.02115) | ~1.06M traj; **full set "Coming Soon"**; only [rbm-1m-ood](https://huggingface.co/datasets/robometer/rbm-1m-ood) is out (571 ep, 161 MB) | real + human video | 21 platforms (OXE, AgiBot…) | Per-frame progress, success and trajectory preference. Failures are partly real suboptimal runs and partly synthetic rewind. OOD set has `quality_label` and `partial_success`. | OOD set Apache-2.0 **OK**; full set ? | prog, succ, pref | RBM-EVAL-ID/OOD (eval only) |
| [DenseReward](https://huggingface.co/datasets/densereward/DenseReward) | 2026-07 (2607.13033) | 27k ep, 1.12M rows, 21.2 GB in 4 zips (droid/isaac/robosuite/libero), downloadable separately | **mostly sim** + DROID | Franka (sim), DROID | Per-frame 0–1 reward, programmatic. Failures synthesised by perturbation: miss, collision, fall, recovery. | Apache-2.0 **OK** | prog, adv, fail | test 56k rows |
| [ProcCorpus-60M](https://huggingface.co/datasets/ce-amtic/ProcCorpus-60M) (ProcVLM) | 2026-05 (2605.08774) | ~400k traj, 60M frames, from 30 sources; annotations 37.7 GB split per source and per type | real + some sim | Bridge, DROID, Fractal, … | Per-frame subtask stage, completion, remaining actions and progress. **All written by a VLM.** | CC-BY-4.0 **OK, G**; source licences still apply | stage, prog, val | none formal |
| [VLAC-Cut-FullData](https://huggingface.co/datasets/InternRobotics/VLAC-Cut-FullData) / [Benchmark](https://huggingface.co/datasets/InternRobotics/VLAC-Cut-Benchmark) (HELP, 2607.09776) | 2026-07 | 23k ep, 375k progress points; 957 GB (+104 GB bench); `--include` works | real + sim | DROID Franka, in-house ARX-5, LIBERO/VLABench | **Human** signed per-frame progress (negative = worse than start) with an explanation per value. Covers advance, stall, regression and recovery. | "other", **no terms written (?)**; code MIT | prog, adv, fail, val | VPB test splits (~3.5k records); **do not train** |
| [Robo-Dopamine GRM](https://huggingface.co/datasets/tanhuajie2001/Robo-Dopamine-GRM-Dataset) | 2026-05 (2512.23703) | 34.6M samples (3.0M balanced), 263 GB, 18 sources | real + sim | AgiBot, DROID, EgoDex, RoboTwin… | Relative "hop" progress between frame pairs, computed from frame index in successful demos. Regressions come from reversed pairs (**synthetic**). | Apache-2.0 **OK, G**; source licences still apply | adv, prog | Robo-Dopamine-Bench (unverified) |
| [RoboReward](https://huggingface.co/datasets/teetone/RoboReward) | 2026-01 (2601.00675) | 54k examples, 43.2 GB | real | OXE + RoboArena, 14 embodiments in bench | One episode-end 1–5 score. Negatives made synthetically (wrong instruction, truncation). RoboArena human 0–100 scores are mapped to 1–5. | CC-BY-4.0 **OK** | succ, prog (coarse) | **test 2,831 = RoboRewardBench (human-verified)** |
| [PRIMO R1](https://huggingface.co/collections/LeonOverload/primo-r1) | 2026-03 (2603.15600) | SFT json 2.42 GB; ~6.6 TB with video | real + sim | AgiBot, BEHAVIOR-1K, RoboTwin, RoboVQA | 0–100 progress with chain-of-thought text. Whether that text is human-written is unverified. | Apache-2.0 **OK**, but its AgiBot frames are NC upstream | prog | bench agibot/behavior/robotwin ID+OOD, real-humanoid-OOD |
| WARP-RM (2606.28320) | 2026-06 | only sim traces released; the real 140 h T-shirt data is **not released** | sim | — | Self-supervised progress velocity (synthetic) | MIT (code) | — | — |
| GVL, VLAC original, SARM paper data, SARM2 | — | **not released** | — | — | — | — | — | — |

### 2. Stage / subtask boundaries

| Dataset | Date | Size / partial | Real? | Embodiments | Labels (provenance) | Licence | Questions | Held-out |
|---|---|---|---|---|---|---|---|---|
| [lerobot/FineART](https://huggingface.co/datasets/lerobot/FineART) | ~2026-10 | 40,543 ep, 1,718 h, 185M frames, 1.86 TB; load by `episodes=[…]` or stream | real | Trossen stationary ALOHA2-style bimanual | **Human** contiguous subtask segments with free text, by annotators separate from the teleoperators, audited by a third team. Episode summary plus quality/success metadata. | CC-BY-4.0 **OK** | stage, prog (time-in-stage), succ | none; make your own by `episode_index` |
| [AgiBotWorld2026](https://huggingface.co/datasets/agibot-world/AgiBotWorld2026) | 2026 | 13.6 TB, per date and task folder | real | AgiBot G2 | Subtask and skill intervals, `is_result_succeed`, `error_cause`, `restorable`, intervention frames. A third party counts ~5.8k error segments and ~10.7k intervention segments. | **CC-BY-NC-SA-4.0 (NC)** | stage, succ, fail, adv | train only |
| [AgiBot World Alpha/Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) | 2024-12 / 2025-03 | Beta 1M+ traj, 48 TB, per task | real | AgiBot dual-arm | `action_config` skill segments (87 skills). About 1% of episodes are failure-recovery, with human-labelled reason and timestamps. | **CC-BY-NC-SA-4.0 + community licence (NC, G)** | stage, fail | — |
| [Galaxea Open-World](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) | 2025-08 | 500+ h, 2.87 TB, 227 task archives | real | Galaxea R1-Lite mobile bimanual | Human subtask instruction with start/end times; `quality_index` | **CC-BY-NC-SA-4.0 (NC, G)** | stage | — |
| [RoboInter-Data](https://github.com/InternRobotics/RoboInter) | 2026-02 | 230k ep (DROID + RH20T re-labelled) | real | Franka + RH20T rigs | Per-frame subtask clip, 15 primitive skills, boxes. Tool-assisted; human vs model share unverified. | **CC-BY-NC-SA-4.0 (NC, G)** | stage | — |
| [RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) | 2024-12 | 107k traj, 12.3 TB, per robot and task | real | Franka, Tien Kung, AgileX, UR5e | Frame-level language. `failure_data/` holds ~5k real failures with causes (e.g. "Object Drop"); the file layout is unverified. | Apache-2.0 **OK, G** (read the user agreement) | stage (coarse), succ, fail | train/val folders |
| [ECoT Bridge](https://huggingface.co/datasets/Embodied-CoT/embodied_features_bridge) | 2024 | all of Bridge V2 (~60k traj) | real | WidowX | Per-step plan, subtask and move labels, **model-generated** (Gemini + detectors) | MIT; Bridge frames CC-BY-4.0 **OK** | stage | — |
| [RH20T](https://rh20t.github.io/) | 2023 | 110k seq | real | 7 rigs | Task text + 0–9 success rating | **Split: scenes 1–5 CC-BY-SA-4.0 OK; scenes 6–10 CC-BY-NC (NC)** | succ (graded) | — |
| [BEHAVIOR-1K demos](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) | 2025/2026 | 10k / 20k ep, ~2 TB | **sim** | R1Pro | Skill and primitive segments, multi-level language | MIT **OK** | stage | challenge eval |
| Fourier ActionNet, EgoDex | 2025 | 2.7 TB / 2 TB | real / human video | GR1/GR2; human hands | Task-level only | **NC** (EgoDex also no derivatives) | — | EgoDex frozen test |
| LeRobot `subtask_index` community sets, SARM tagged sets (e.g. `lerobot-data-collection/three-folds-embed-sarm`, 273 ep, Apache-2.0) | 2026 | small | real | SO-101, OpenArms | Human or VLM subtask columns; mostly unverified | Apache-2.0 (often the HF default) or none | stage | — |

### 3. Value / advantage / success / interventions

| Dataset | Date | Size / partial | Real? | Embodiments | Labels (provenance) | Licence | Questions | Held-out |
|---|---|---|---|---|---|---|---|---|
| [DROID](https://droid-dataset.github.io) ([lerobot/droid_1.0.1](https://huggingface.co/datasets/cadene/droid_1.0.1)) | 2024 | 95.6k ep (~76k success, ~16k failed), 27.6M frames, ~400 GB | real | Franka | Operator-set `is_episode_successful`; 3 language instructions per episode. Failures are failed demonstrations. | CC-BY-4.0 per the paper; LeRobot card Apache-2.0 **OK** | succ, val (time-to-go on successes) | — |
| [RoboArena dumps](https://huggingface.co/datasets/RoboArena/DataDump_02-03-2026) | 2025-08 / 2026-02 | 23.6k rows, 18.5 GB (+8.1 GB older) | real | DROID Franka, 7+ sites | **Human** A/B/tie preference, 0–100 progress per rollout, free-text reason, 3 cameras | MIT **OK** | pref, prog, succ | none (RoboReward test overlaps; dedupe) |
| [robot-policy-human-preferences-55k](https://huggingface.co/datasets/datapointai/robot-policy-human-preferences-55k) | 2026 | 5,494 pairs × 10 crowd votes, 2,861 rollouts, 27.5 h | real | bimanual YAM | **Human** crowd preferences, which give soft targets | votes CC-BY-4.0, clips Apache-2.0 **OK, G** | pref | **cup_stacking task held out** |
| [BodenAI RW-RL-HIL](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset) | 2026-09 | 3,347 deployment ep (3,068 with an intervention), 82 h | real | Galaxea R1Lite | Per-frame control-authority flag, intervention segments | CC-BY-4.0 **OK** | adv (pre-intervention = negative), fail-onset proxy | — |
| [MINT-SJTU RW-RL](https://huggingface.co/datasets/MINT-SJTU/RW-RL-Dataset) | 2026-10 | 24.5M frames, 5.52 TB, per task | real | R1Lite, G2A, SO101, Piper | reward, done, completion, intervention start/end, model-assisted rank and stage scores | **none stated (?)** | val, adv, stage, succ | — |
| [Robo-ValueRL](https://huggingface.co/datasets/X-Humanoid/Robo-ValueRL) | 2026-07 (2607.09866) | 431 GB, 2 task folders | real | Tienkung dual-arm humanoid | Per-frame `remain_time` and GOOD/MEDIUM/BAD quality; mixes success, correction and failure (programmatic) | **none stated (?)** | val, adv, succ | — |
| [Sirius-Fleet MUTEX](https://huggingface.co/datasets/huihanl/sirius-fleet-mutex) | 2024 | 51.9 GB; card empty | real | Franka | `intv_labels` human interventions | CC-BY-4.0 **OK** | adv | — |
| [TRANSIC](https://huggingface.co/datasets/transic-robot/data) | 2024 | 11.2 GB | real + sim | Franka | Human corrections, pre/post intervention | MIT **OK** | adv | — |
| [RoboChallenge Table30v2](https://huggingface.co/datasets/RoboChallenge/Table30v2) | 2025-10 | 1.01 TB, per task | real | ARX5, UR5, ALOHA, DOS-W1 | Demonstrations only; scored evaluation rollouts not confirmed downloadable | **none (?)** | — | benchmark |
| [ManipArena](https://huggingface.co/datasets/ManipArena/maniparena-dataset) | 2026-03 | 10.8k demos, 253 GB | real + sim | bimanual | **No score labels**; partial credit lives only in the evaluation protocol | Apache-2.0 **OK, G** | — | eval rollouts unreleased |
| OXE subsets: `kuka`, `fractal`, `bc_z`, `mt_opt` | 2018–2023 | per-subset RLDS | real | KUKA, Everyday Robots | Real binary success, including failures; `bc_z` adds intervention/autonomous flags | per subset, mostly CC-BY-4.0 / Apache | succ, adv | — |
| FurnitureBench | 2023 | 5,100 demos, 219 h | real | Franka | Sparse +1 per assembled part (auto, then human-corrected) | data licence not found (?) | stage, val | — |
| π*0.6 / RECAP, ConRFT, RL-100 real, AutoEval | — | **no public data** | | | | | | |

### 4. Failure detection / recovery

| Dataset | Date | Size / partial | Real? | Embodiments | Labels (provenance) | Licence | Questions | Held-out |
|---|---|---|---|---|---|---|---|---|
| [FailTime](https://huggingface.co/datasets/ChangUoA/FailTime) | 2026-09 (2609.21369) | 6.1 GB; Short = 1,539 ep (annotations over ViFailback video); Long = 100 ep | real | AgileX Piper, ALOHA, SO-101 | Success/fail, **failure onset time (s)**, type, expected-vs-observed reason, avoidance. Human per the card. | CC-BY-4.0 **OK** | fail, succ | **FailTime-Long is test only** |
| [ViFailback](https://huggingface.co/datasets/sii-rhos-ai/ViFailback-Dataset) | CVPR 2026 | 5,202 traj (4,545 failures), 58k QA, 513 GB | real | ALOHA-type / Piper | **Human** failure keyframe, 4 failure types, visual correction hints | MIT **OK** | fail, succ | not stated; avoid any FailTime-Long overlap |
| [REBOOT](https://huggingface.co/REBOOT26) | 2026-09 (2609.22591) | 2,160 demos (1,080 expert + 1,080 recovery), 38 LeRobot sets | real | bimanual Trossen WidowX AI | Which of 5 phases failed, failure mode (misalignment, jamming, premature release), paired recovery. Failures were **staged deliberately** to mimic policy failures. | HF tags Apache-2.0; project page CC-BY-4.0 **OK** (the paper itself is CC BY-NC-SA) | fail, stage, adv | — |
| [Guardian / FailCoT](https://www.di.ens.fr/willow/research/guardian/) (2512.01946) | 2025-12 | RLBench-Fail (sim, 63k samples); BridgeDataV2-Fail (real, 37k samples, perturbed); UR5-Fail (real policy rollouts, 2.8k samples) | mixed | WidowX, UR5 | 0/1 reward, planning vs execution fault, failure mode, chain-of-thought | UR5-Fail Apache-2.0 **OK**; other cards unread (?) | succ, fail | val/test per part |
| [RoboFAC](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset) | 2025-05 | 10k videos (480 real failures), 78k QA, 25.8 GB | **mostly sim** | ManiSkill + real | Failure type and correction QA | MIT **OK** | fail | — |
| [RACER](https://huggingface.co/datasets/sled-umich/RACER-augmented_rlbench) | 2025 | 10k ep, 129 GB | sim | RLBench Franka | Perturbation failures with GPT-4 recovery language | MIT | fail | val |
| AHA / FailGen | 2024-10 | generate it yourself (49k pairs) | sim | RLBench | Perturbation failures | **NVIDIA licence, research only (NC)** | — | — |
| RoboFail (REFLECT) | 2023 | 30 real + 100 sim | mixed | UR5e | Failure explanations | not found (?) | fail | — |

## Real within-trajectory setbacks (progress lost)

No public dataset is all four of: real, human-labelled, commercially licensed, and labelled per frame with progress *dropping* (drops, slips, undone steps). The closest:

1. **VLAC-Cut-FullData.** This is the only verified release with human, signed per-frame progress on real ARX runs. The paper says those runs "intentionally include failed, regressive, and recovery trajectories": drops, re-grasps, wrong object. Its licence is "other" with no terms written down, so we need written permission from InternRobotics.
2. **AgiBotWorld2026 / Beta.** Real, human error frames with `error_cause`, `restorable` and intervention segments. **Non-commercial.**
3. **ViFailback, FailTime, REBOOT.** Real failure keyframes or onset plus recovery. All are commercial-OK. They give a "failure began here" label, not a progress curve, but we can derive a negative-advantage target across the onset.
4. **RoboMIND `failure_data/`.** Real "Object Drop" class causes, episode-level only. Apache-2.0, gated.
5. **Synthetic only:** DenseReward (fall/recovery), Robo-Dopamine (reversed pairs), Robometer rewind, RoboReward truncation.

The HELP paper's in-house ARX data is the right shape, but beyond what is in VLAC-Cut it is unreleased. **Our own floor data is the realistic source for this class.**

## Ranked shortlist (commercial, real, dense, diverse)

Raw sizes add up to well over 2.6 TB, so the budget below takes subsets. Prefer lower-resolution or single-camera streams and decode only the frames we sample.

| # | Source | Why | Recommended subset | Budget |
|---|---|---|---|---|
| 1 | **lerobot/FineART** | Largest human-labelled real subtask set under CC-BY-4.0, with success/quality metadata | ~8–10k episodes stratified by task, metadata for all; hold out ~5% of episodes | ~450 GB |
| 2 | **RoboArena dumps (both)** | Real human pairwise preference + 0–100 progress, MIT | all | 27 GB |
| 3 | **DROID (lerobot/droid_1.0.1)** | Real success *and* failure, diverse scenes; base for time-to-go value | all ~16k failures + ~20k successes, 1–2 cameras | ~150 GB |
| 4 | **ViFailback + FailTime** | Real human failure keyframes and onset times | FailTime all (6 GB); ViFailback annotations + ~2k trajectories | ~200 GB |
| 5 | **REBOOT** | Real failure phase/mode + paired recovery on precision assembly | all 38 sets | ~50 GB (estimate) |
| 6 | **BodenAI RW-RL-HIL + Sirius-Fleet MUTEX + TRANSIC** | Real interventions; the pre-intervention window gives negative advantage | all | ~150 GB (estimate) |
| 7 | **RoboReward (train/val only)** | 1–5 episode scores on real OXE/RoboArena data | train + val; **keep test as eval** | 43 GB |
| 8 | **datapointai preferences-55k** | 10 votes per pair give real soft preference targets | all (accept the gate) | <50 GB (estimate) |
| 9 | **ProcCorpus-60M annotations** | Dense stage + progress over many real sources; model labels, so use as weak/soft targets | annotations only (37.7 GB) joined to frames we already hold (DROID, Bridge) | ~40 GB + reuse |
| 10 | **RoboMIND (failure_data + a subset)** | Real failure causes across 4 robots, Apache-2.0 | `failure_data/` + ~10 tasks | ~400 GB |

**Total ≈ 1.6 TB.** That leaves about 1 TB of headroom for DenseReward (21 GB, sim augmentation for setbacks), Robo-Dopamine `json_sel` + images (≤263 GB, synthetic pair advantage) and decoded frame caches.

**Pending terms:** if InternRobotics gives written permission, add VLAC-Cut (the test splits plus the ARX subset first, ~150 GB). It is the single best setback source. The same applies to Robo-ValueRL and MINT-SJTU RW-RL once their owners state a licence.

## Open risks

- **Non-commercial licences cover the richest stage and error labels:** AgiBot (all versions), Galaxea, RoboInter, Fourier, EgoDex, AHA and RH20T scenes 6–10. Do not ingest them, and do not ingest their derivatives either: PRIMO and Robo-Dopamine include AgiBot frames, so filter those sources out by name.
- **Unlicensed is not open:** VLAC-Cut ("other"), Robo-ValueRL, MINT-SJTU RW-RL and RoboChallenge state no terms. Get written terms first.
- **Synthetic negatives:** most dense-progress sets build failures by rewind, truncation or perturbation. Those train "reversed video looks bad", not real slips. Weight them down and keep real failures as the eval set.
- **Model-generated labels:** ProcCorpus, ECoT and SARM's LeRobot annotations are VLM-written. Use them as soft targets with lower weight, never as eval.
- **Held-out contamination to avoid:**
  - RoboRewardBench (RoboReward test)
  - FailTime-Long
  - VLAC-Cut VPB test
  - RBM-EVAL-ID/OOD
  - PRIMO bench
  - datapointai cup_stacking
  - DenseReward test split
- **Overlap:**
  - RoboReward and RoboArena share rollouts.
  - FailTime-Short reuses ViFailback videos.
  - VLAC-Cut, RoboInter and ProcCorpus all re-label DROID.

  Dedupe by source episode id before splitting, or the eval leaks.
- **Gated sets:** ProcCorpus, Robo-Dopamine, RoboMIND, datapointai and ManipArena each require accepting terms per account. Record who accepted and which version.
- **Unverified items:**
  - FineART exact release date
  - PRIMO human-vs-model rationale
  - RoboMIND failure-file layout
  - Guardian's Bridge-Fail licence
  - Sirius-Fleet trajectory count
  - DROID licence: CC-BY-4.0 per the paper, not a repo LICENSE file
  - REBOOT licence: Apache-2.0 tag vs CC-BY-4.0 on the page; both are permissive
