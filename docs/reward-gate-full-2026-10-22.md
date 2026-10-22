# Full laptop reward gate — 22 October 2026

**Decision: do not advance to paid labeling or Modal on this evidence.** Spatial pooling won this one-seed outcome screen. Temporal late interaction failed the predeclared 0.02 MAE improvement gate, and the fixed ABC pack supplied no visually qualified physical mistake/recovery events. Neither result disproves a low-dimensional recovery representation; this run does not establish one.

## Completed experiment

The Apple M1 Max laptop (64 GiB unified memory, PyTorch MPS) completed **8,001 training examples and 514 development examples**. Whole source families explain the slight overshoot of the requested 8,000/512 counts. Training covers 1,846 families; development covers 125. Seed 17, three epochs, 378 optimizer updates per head. All three controls use the same observations, instructions, membership, frozen encoder and training recipe.

Frozen encoder: `google/siglip2-base-patch16-224` at `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. Eight uniformly timed video observations retain all 196 image patches and contextual text tokens. FP32 encoder inference produces BF16 cache values. Lossless storage compression does not change those values. The 375,187,970-parameter encoder is additional to each small trainable head; this is not a wholly sub-million-parameter perception system.

AdamW uses learning rate 3e-4, weight decay .01, gradient clipping 1, microbatch 8 and effective batch 64. Loss is categorical cross-entropy plus .25 normalized ordinal ranked-probability loss. The recipe was frozen before training in `execution-plan.json`; no head was tuned after seeing these results. Code revision: `d9d8d014988a1fe327f86fdfffb28f36249d8fa9`.

## Outcome results

Lower MAE and Brier are better. Accuracy/Brier below are from the epoch selected by development MAE.

| Head | Parameters | Best epoch | Best dev MAE | Final epoch MAE | Accuracy | Brier | Head run time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `late_temporal` | 627,845 | 3 | 1.1089 | 1.1089 | 46.89% | 0.6919 | 176.7s |
| `pooled_temporal` | 627,589 | 3 | 1.0506 | 1.0506 | 50.19% | 0.6478 | 128.5s |
| `late_orderless` | 626,821 | 2 | 1.1381 | 1.1829 | 46.50% | 0.6577 | 132.0s |

The training-median constant predicts reward 3 and obtains **1.3872 MAE** on development. All three heads improve on that simple point estimate, but late interaction adds no demonstrated advantage over pooling.

Primary difference is `pooled MAE - late_temporal MAE`, so positive values favor late interaction:

- **−0.0584 MAE**, versus the required **+0.0200**. Outcome gate: **FAIL**.
- Paired 95% episode-family bootstrap interval: **[−0.1603, +0.0454]**, 10,000 draws, seed 17, resampling all rows of a family together.
- Pooling has the better point estimate, but the interval crosses zero; this does not prove pooling is universally superior.
- The primary comparison is identical at the fixed third epoch because both temporal heads selected epoch 3.
- Temporal versus orderless late interaction: best-epoch improvement **+0.0292**, interval **[−0.1137, +0.1653]**. Fixed-final-epoch improvement **+0.0739**, interval **[−0.0225, +0.1697]**. Neither establishes a temporal advantage.

These are conditional, one-seed development estimates, not final test results or training-seed uncertainty. Best-epoch estimates also involve development selection. Native StartLux and FLORA were not run, so no reference-model superiority is claimed. No RLCD or mistake/recovery objective was trained.

The input checks behave as intended: task/video swaps change all heads' outputs. Reversing frames changes the two temporal heads, while the orderless head's probability delta is exactly zero on the eight-example probe. These are mechanics checks, not semantic failure labels.

## Leakage and cache integrity

RoboReward is pinned to `469b9af76e8539e8d2ac553b081307738fd92ca5`. The first full selection was rejected before training: exact video bytes crossed training/development under different episode names. Selection now quarantines entire families using both source lineage and the pinned full-corpus video hashes, reserving validation/test material before minting membership. **24 training rows were excluded** before reselection. No official test `metadata.jsonl` or rewards were downloaded; only the video file inventory and LFS content hashes were used for quarantine.

The final manifest digest is `376b9f9117b51542022be22ae5c76b50b4738a07e98dfb0321918657f2b83321`. Selected train/dev media hashes are disjoint, and there are no identical-video-plus-instruction groups with conflicting labels in either split. This is source-family separation, not a claim of task-disjointness, perceptual near-duplicate removal or absence from encoder pretraining.

The complete cache contains **5,908 unique video encodings and 3,828 unique instruction encodings**, occupying **10,584,987,044 bytes (10.585 GB)**. It reuses 73 visual and 84 text encodings from the earlier smoke. All 157 legacy files were independently checked against their original hashes; future reuse also enforces that check. Every final indexed file is verified before training, and each optimizer group is checkpointed. The deliberate encoding interruption changed only the lossless compression level and resumed durable progress; it did not change the encoder or observation budget.

Encoder specification digest: `c08f28882fa24757d9e5285f784eb8b45a8f82a1e2944c9bbced3c410e1c88ce`. The rejected first manifest and prior smoke remain historical artifacts, not the inputs to this run.

## ABC audit result

The [complete 30-window report](reward-gate-abc-audit-2026-10-22.md) records all judgments and limits. The fixed pack covers ten episodes/tasks, with ten keyword, ten repetition and ten ordinary-control windows. All 150 contact sheets were inspected using top and both wrist views at eight uniform times and then 32 denser times per 16-second window.

- **0** clear mistake-only events, **0** failed corrections, **0** successful restorations.
- **19 ordinary-visible**, **11 unknown**; ordinary does not mean a certified error-free negative.
- Required usable-event count: 10/30. Observed: **0/30**. ABC gate: **FAIL**.
- These are dependent, selected windows at thumbnail/sampling resolution. Occlusion, tiny contact regions, missing pre-event context and ambiguous regrasp intent prevented qualification. The audit is model-assisted, not independent human gold, and does not show that ABC contains no recoveries.
- Every row remains `training_eligible=false`, `human_label=null`. No ABC hint entered head training.

## Actual local cost and retained artifacts

- Encoder pipeline: **48.29 minutes**, including startup, bounded media transfer, decode, compression and one resume. Pinned model weights were already cached.
- Three-head stage, including cache verification/loading: **8.19 minutes**. Individual head times are in the table. Combined encoder-plus-training stages: **56.48 minutes**.
- ABC CPU extraction: **10.84 minutes**, overlapping encoder work; 2,482,209,052 source bytes passed through RAM and 43,145,959 JPEG bytes were retained.
- **Modal/other rented compute: $0. Paid labeling: $0.** Laptop power and existing Codex usage were not metered as a new experiment charge.
- Source videos were streamed in RAM rather than stored as another full copy. All features, predictions, checkpoints, manifests and audit sheets remain on this laptop.

The run artifacts are in ignored `runs/reward-gate/full-8000-clean/`: `completion-receipt.json`, `execution-plan.json`, `dataset-audit.json`, `encoding-receipt.json`, `cache/index.json`, and `training/`. Each head retains `last.pt`, `best.pt`, `receipt.json`, `predictions.json` and `last_predictions.json`. `training/comparison.json` holds the gate result; `training/uncertainty.json` holds paired intervals and source-level MAEs. The completion receipt binds the evidence files by SHA256. ABC provenance and sheets are in `runs/reward-gate/abc-full-audit/`.

[Reproduction commands](reward-gate.md) use this exact full-run manifest/output directory. Re-running training with unchanged inputs/code/config resumes completed checkpoints. The existing native StartLux inference server is unchanged.

## Decision boundary

Keep spatial pooling as the local baseline and preserve this cache. This screen supplies no reason to buy larger compute or manufacture recovery labels from the current ABC hints. A future continuation would need either repeatable late-interaction gains across training seeds or a separately qualified physical-event dataset/observation protocol. Those follow-ups were not performed as part of this gate.

Validation: the 282-test repository suite passed (18 skipped); additional paired-bootstrap and legacy-reuse corruption regressions passed. Faster-compression roundtrip tests passed; Ruff and whitespace checks passed. Actual full-data execution, rather than those software tests, supplies the experiment results above.
