# Small reward-model initial gate

This offline experiment trains three outcome heads on frozen SigLIP 2 features from
RoboReward and mines ABC annotation intervals for review. It does not train a recovery
classifier, run RL, invoke StartLux/FLORA as judges, launch cloud compute, or authorize
robot motion. The existing inference server is unchanged.

The [full 8k laptop report](reward-gate-full-2026-10-22.md) records the completed outcome
screen and its failed continuation gate. The [30-window ABC audit](reward-gate-abc-audit-2026-10-22.md)
also failed its usable-event gate. The earlier [laptop smoke report](reward-gate-smoke-2026-10-22.md)
remains historical software qualification.

## Install and run

Python 3.12 is recommended. Dependencies stay in the optional extra:

```bash
uv venv --python 3.12
uv pip install -e '.[reward-gate]'
```

A bounded local smoke (96 training records and at least 32 development records,
keeping entire source-episode families) is the default. Run from the checkout:

```bash
python -m startlux_decide.reward_gate prepare \
  --manifest runs/reward-gate/roboreward/manifest.json
python -m startlux_decide.reward_gate cache \
  --manifest runs/reward-gate/roboreward/manifest.json \
  --output runs/reward-gate/roboreward/cache \
  --device mps --media-budget-mb 128 --feature-budget-mb 512
python -m startlux_decide.reward_gate train \
  --manifest runs/reward-gate/roboreward/manifest.json \
  --cache runs/reward-gate/roboreward/cache \
  --output runs/reward-gate/roboreward/training --device mps
python -m startlux_decide.reward_gate inspect-abc \
  --output runs/reward-gate/abc --max-episodes 30
```

Use `--device cuda` on a GPU worker or `cpu` for a CPU check. After installation,
`startlux-reward-gate` is an equivalent entry point. `cache --local-model` forbids
encoder downloads. The encoder download itself needs about 1.54 GB; media and feature
budgets are additional. Download/cache operations retain a 512 MiB disk reserve.
No command accepts dataset terms or purchases remote compute. Existing Hugging Face
credentials are used for gated ABC access and never written into outputs.

For the full outcome gate, prepare a **new manifest/output directory** with
`--train-count 8000 --dev-count 512`. Estimate media sizes and raw feature storage first,
then set explicit budgets suitable for that worker. Do not reuse or overwrite a smoke
manifest. The small local run establishes software behavior; its `gate_passed` is null.
For at least 8,000 training and 512 development records with three epochs, the outcome
continuation threshold is a 0.02 improvement in MAE versus spatial pooling. ABC visual
review is a separate requirement. This implementation does not claim that weak action
annotations or recovery supervision have been added to outcome training.

The full laptop run uses streamed media and lossless compression. Compression preserves
the exact cached tensor bytes; it does not pool patches or reduce precision. At most eight
bounded media downloads are prefetched in RAM. Compatible prior frozen features can be
reused without re-encoding, including when the new manifest has different membership.

```bash
python -m startlux_decide.reward_gate prepare \
  --manifest runs/reward-gate/full-8000-clean/manifest.json \
  --train-count 8000 --dev-count 512
python -m startlux_decide.reward_gate cache \
  --manifest runs/reward-gate/full-8000-clean/manifest.json \
  --output runs/reward-gate/full-8000-clean/cache \
  --device mps --local-model --stream-media --compressed \
  --reuse runs/reward-gate/roboreward/cache \
  --media-budget-mb 12000 --feature-budget-mb 16000
python -m startlux_decide.reward_gate train \
  --manifest runs/reward-gate/full-8000-clean/manifest.json \
  --cache runs/reward-gate/full-8000-clean/cache \
  --output runs/reward-gate/full-8000-clean/training --device mps
python -m startlux_decide.reward_gate analyze \
  --manifest runs/reward-gate/full-8000-clean/manifest.json \
  --training runs/reward-gate/full-8000-clean/training
```

The analysis pairs predictions by record ID and resamples whole episode families 10,000
times. It reports both best-development-epoch and fixed-final-epoch MAE differences.
These intervals describe a one-seed development screen; best-epoch selection remains
optimistic, and neither interval establishes superiority over a reference model.

## Frozen encoder and controls

Encoder: `google/siglip2-base-patch16-224`, revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The image tower returns 196×768 patch
features per image. Text uses contextual token features, a fixed maximum of 64 tokens,
and an attention mask. Overlength instructions fail explicitly; they are not silently
truncated. Both towers run in evaluation/inference mode with every parameter frozen.

Eight nearest uniform presentation timestamps include both video endpoints. The cache
holds BF16 features before trainable projection, original sampled seconds and relative
times. The forward pass uses FP32. Visual features are shared across instructions for the
same source bytes; text features are shared across identical tasks. This avoids repeated
encoder work when comparing heads.

| Head | What it retains | Trainable parameters |
|---|---|---:|
| `late_temporal` | Query-to-patch matching, spatial coordinates and temporal attention | 627,845 |
| `pooled_temporal` | Per-view spatial pooling, text and the same temporal attention | 627,589 |
| `late_orderless` | Query-to-patch matching, then time pooling and matched-size MLPs | 626,821 |

All project 768-dimensional features to width 128 and output five outcome logits. Temporal
heads use two layers with four attention heads and feed-forward width 512. The orderless
head has no time input to its computation. Masked cameras/text cannot influence evidence.
The three sizes differ by less than 0.2%; all counted weights are used. The encoder is
additional: the pretrained checkpoint has 375,187,970 parameters, not 0.7M.

Training: AdamW, learning rate 3e-4, weight decay .01, gradient clipping 1, microbatch 8,
effective batch 64, seed 17, three passes. Loss is categorical cross-entropy plus .25
normalized ordinal ranked-probability loss. Labels 1–5 convert once to classes 0–4.
Each head selects its best epoch by development discrete-score MAE. Reports retain all
epochs, per-example five-class probabilities, Brier score and accuracy. No calibration
or final benchmark claim is made at this stage.

## Integrity and resume

- RoboReward is pinned to `469b9af76e8539e8d2ac553b081307738fd92ca5`.
  Original OXE episode indices and RoboArena recording UUIDs group clips, relabelings
  and camera views. Unknown lineage fails closed. All official validation/test families
  are reserved before training selection; development selection also excludes test
  families. Pinned LFS content hashes for the whole video inventory also quarantine
  families whose clips occur under different names in reserved splits. Official test
  metadata/labels are never downloaded.
- Only task text and observations enter a head. Neither score-bearing filenames nor
  `gpt5_mini_check` rationales enter features. Metadata hashes and exact membership are
  saved in an immutable manifest. Same-byte media crossing train/dev is refused.
  Full-corpus perceptual near-duplicate discovery and unknown pretraining exposure
  remain unaddressed; do not claim stronger decontamination.
- Feature identities bind the encoder revision/processor/runtime, media content hash or
  exact query, timestamp sampling and dtype. The index binds each record to its expected
  query/media identity and feature-file SHA-256. Corruption, reassignment, changed
  encoders and incomplete caches fail before training.
- Each feature and completed optimizer group is saved durably. Checkpoints contain
  weights, optimizer, sampler cursor and CPU/accelerator/Python RNG state. Rerunning the
  **same command** resumes; changed code/config/cache is refused. `--max-updates 1`
  provides a deliberate interruption boundary for a resume smoke.
- Run directories and weights are ignored by Git. Never delete source `raw/` material.
  Local receipts record wall time and file digests; they are not ERP bookings or managed
  Job receipts. A paid remote run must use the existing managed Job and spend controls.

Cache/training commands are single-writer per output directory. Use distinct output
paths for concurrent runs. Do not run two writers against the same cache/index.

## ABC candidates: hints awaiting visual review

ABC is pinned to `d4c79f5aecb902f3157e6139661cb1b0b2a36689`. The inspector only reads
`data/train/.../annotation.mcap`, bounded by episode, byte and listing limits. It parses
the actual self-describing protobuf `Annotation {timestamp, data}` on
`/subtask-annotation`. Times are absolute integer nanoseconds. Interval ends come from
the next transition; the final end stays unknown without the episode file.

Nonconsecutive repeated action text and correction/mistake/reset words produce candidate
intervals. Consecutive identical transitions are merged. Ordinary non-keyword,
nonrepeating intervals provide review controls. Repetition can be ordinary task structure,
and "reset both arms" can be an episode terminator. Every exported interval retains
`verification_status=unverified`, `training_eligible=false`, and `human_label=null`.

`report.json` contains provenance, action strings, timestamps, reasons, advertised episode
sizes, candidate counts and access errors. Counts are candidate yield, not event-label
precision. A model-assisted contact-sheet inspection is also not independent human gold.
Before buying labels, select a balanced 30-window review pack, verify visible mistakes,
failed corrections and restorations of the same object/condition, and audit the eight-frame
observation budget. The ≥10/30 usable-event gate cannot be passed from annotations alone.

## Verification

```bash
python -m unittest discover -s tests -p 'test_reward_gate_*.py' -v
python -m unittest -v
```

The focused tests cover family quarantine, altered manifests, cache corruption and
query/media reassignment, frozen encoder gradients, timestamp sampling, masking,
spatial/temporal controls, real MCAP parsing, and exact optimizer/RNG resume equivalence.
Tests using locally initialized encoders or generated videos establish mechanics; a live
pinned-checkpoint run is recorded separately in the implementation report.
