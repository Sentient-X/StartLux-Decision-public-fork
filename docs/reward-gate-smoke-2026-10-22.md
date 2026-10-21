# Initial gate implementation and laptop smoke — 22 October 2026

**Implemented and exercised locally on Apple M1 Max, 64 GiB RAM, using PyTorch MPS.**
No Modal job, other rented GPU, model API call, or paid labeling work was launched.
This is a software/measurement smoke, not the full 8,000-example outcome gate or a
mistake/recovery benchmark result.

## Real-data execution

Source code starts at StartLux revision `83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01`.
The [run instructions](reward-gate.md) document exact dataset/encoder revisions and commands.

- Requested 96 training and 32 development records. Whole-family selection produced
  **96 training + 33 development = 129 records**, from 26 source families (19 train,
  7 development). No selected source family overlaps the official test inventory.
  Test instructions/rewards were not downloaded.
- Manifest digest:
  `da25b1bed19adda1b74e5278a5c73d14729c83b48de9f13687d6019c904680b4`.
- Real pretrained SigLIP encoded **73 unique videos × 8 frames** and **84 distinct tasks**.
  Shared videos/tasks reused features. The cache occupies **184,164,672 bytes**.
- Media download/decode/feature loop: **237 seconds**, excluding initial checkpoint
  download/model loading. This is a local throughput observation, not a serving benchmark.
- All three heads performed one update, saved optimizer/RNG/sampler state, stopped, and
  resumed the same run to **six updates / three epochs each**. All weights/probabilities
  remained finite. The encoder was not loaded by the training process.

| Head | Trainable parameters | Best dev MAE (discrete 1–5) | Selected epoch |
|---|---:|---:|---:|
| Temporal late interaction | 627,845 | 1.3333 | 1 |
| Spatial pooling + temporal attention | 627,589 | 1.3939 | 1 |
| Orderless late interaction | 626,821 | 1.6364 | 1 |

The candidate's apparent 0.0606 advantage over pooling comes from only seven development
families and checkpoint selection on that set. **It does not establish a learning gain.**
The receipt sets `eligible_for_outcome_gate=false` and `gate_passed=null`; the official
RoboReward test and StartLux/FLORA reference inference remain untouched.

Input-sensitivity diagnostics on eight development records changed predicted probabilities
when tasks or videos were swapped. Reversing frame order changed temporal-head outputs;
the orderless head's measured probability difference was exactly zero. These are mechanics
checks, not new labels or evidence that the model understands recovery.

## ABC findings before labeling spend

Using existing authorized Hugging Face access, the annotation inspector read **30 train
annotation files, 70,983 bytes**. It exported **539 unverified intervals**: 506 repeat hints
and 74 keyword hints, with overlap between categories. Keyword hints included 44 occurrences
of “mistake” and 30 of “reset.” It also sampled 60 ordinary review controls.

This was a task-shuffled, bounded convenience sample. Repeated task steps are often ordinary
execution; arm resets may be episode termination. Candidate counts are not mistake or
recovery precision. All rows remain `training_eligible=false`, with no human gold labels.

One additional visual spot check downloaded a **64,546,931-byte** locking-task episode,
within a 64 MiB cap and with more than 512 MiB disk reserve. Twelve top-camera frames were
inspected around an annotated **0.500-second “mistake”** and a following **1.799-second
locking attempt**. Manipulation/repositioning is visible, but the contact region is partly
occluded. Neither the specific error nor successful recovery is established.

Consequences for the next decision:

1. Annotation mining works and provides concrete review intervals.
2. Sparse full-episode sampling can miss a half-second event. Local event sampling and
   observability need review independently of this episode-outcome smoke.
3. No ABC hint becomes a recovery target. The planned 30-window visual audit and its
   ≥10 usable-event threshold are **not complete**; paid labeling should wait for that audit.

## Checks and retained evidence

- Full existing/new unittest suite: **274 tests run, 18 skipped, passed**.
- Focused gate suite after final training diagnostics: **29 tests passed**.
- CPU interrupt/resume regression matches uninterrupted weights and predictions exactly;
  stale input/config/cache identities are rejected.
- Tests also cover held-out grouping, cache byte and identity corruption, frozen encoder
  gradients, true presentation-time sampling, padding and temporal/spatial behavior,
  native ABC protobuf decoding and prevention of automatic recovery labels.
- Ruff and whitespace checks passed. No inference-server or native StartLux readout code changed.

Artifacts are retained locally under the ignored `runs/reward-gate/` directory:

```text
roboreward/manifest.json
roboreward/cache/index.json
roboreward/cache/features/*.safetensors
roboreward/cache.log
roboreward/training/{late_temporal,pooled_temporal,late_orderless}/
  last.pt, best.pt, receipt.json, predictions.json
roboreward/training/comparison.json
roboreward/train-interruption.log
roboreward/train-resumed.log
abc/report.json
abc/visual-review.md
abc/visual-frame-manifest.json
abc/visual-contact-sheet.jpg
environment.json
unittest.log
focused-tests.log
```

Observed runtime: torch 2.11.0, transformers 5.13.1, huggingface-hub 1.23.0,
safetensors 0.8.0, Pillow 12.3.0, PyAV 18.0.0, mcap 1.4.0,
mcap-protobuf-support 0.5.4. Training and downloads used the laptop's existing environment;
no change was made to the neighboring `glued` environment or repository.

The local smoke establishes a usable cache/training/candidate-inspection path. It does not
establish the full gate, an advantage over StartLux/FLORA, better downstream policies,
or a qualified robot.
