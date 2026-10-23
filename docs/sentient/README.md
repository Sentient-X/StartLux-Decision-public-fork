# Sentient Decision

StartLux-Decision fine-tuned into a calibrated reward, value and advantage model for robot manipulation. Every
answer is a probability over the options of a typed question about camera frames: how far along the task is, which
step is under way, whether a window moved the task forward, whether something has gone wrong, which rollout is
better.

It is built with the release models' own procedure (technical report, arXiv 2610.17309), not a new method:

| Report | Here |
|---|---|
| Typed decisions with soft targets, several questions per state (Stage 1–2 data) | `sentient/data`: robot episodes become fine-tuning records, each question only where the source's labels support it, targets as sharp as the label behind them |
| Hash-permuted options; broad formats | `sentient/data/formats.py`: state layouts, wording, option names and order, score scales drawn per record |
| Part of the earlier mixture kept in Stage 2 | `sentient/data/retention.py`: general typed decisions from permissive training splits (~20%) |
| Soft cross-entropy on the letter read-out, LoRA adapters merged (Stage 2) | `v1/finetune/finetune_lora.py`, unchanged except that a record may carry images, read exactly as served (`check_image_readout.py`: difference 0.0 against the v1.1 engine) |
| Seed-averaged weights | `sentient/average.py` |
| Temperature per question type, refitted on held-out data | `v1/finetune/calibrate.py` (now reads image records) |
| Reasoning supervision and RLCD² (Stage 3) | later, only if reasoning helps on hard cases |

## Questions

| Kind | Type | Teaches | Needs |
|---|---|---|---|
| progress, remaining | score | value: how far along, how much is left | dense progress |
| time_to_go | score | value: seconds until done | success, episode ends when done |
| will_succeed, succeeded | noul | outcome forecast and verdict | success label |
| end_progress, episode_score | score | partial success, graded quality | end-of-episode human score |
| stage, stage_progress | choice, score | SARM-style stage and progress within it | stage boundaries |
| change, compare | choice | advantage: did the window move the task forward | dense progress |
| failed, failure_type, failure_reason | noul, choice | failure onset and diagnosis | exhaustive failure marks |
| takeover | noul | an imminent human intervention | intervention times |
| preference | choice | which rollout is better | human comparison |

## Sources

See [datasets.md](datasets.md) for the full catalog. In use: FineART (CC-BY-4.0, human subtask segments), FailTime-Short
over ViFailback (CC-BY-4.0 / MIT, human failure onsets, types and reasons), RoboArena (MIT, human success, partial
success and preferences), Robometer OOD (Apache-2.0, test only). Designated test splits never train; families never
straddle splits; six FineART tasks are held out whole.

Licences: only commercially usable data is used. The StartLux-Decision **weights are CC BY-NC 4.0**; commercial use of
a fine-tuned model needs StartLux's agreement.

## Run

```bash
python -m sentient.data.build --source fineart --raw $RAW --out $BUILT/fineart          # per source
python -m sentient.data.retention --out $BUILT/retention
python -m sentient.data.mix --built $BUILT --out $MIX --source fineart --source failtime --source roboarena:15000
python v1/finetune/finetune_lora.py --model $BASE --train $MIX/train.jsonl --dev $MIX/dev.jsonl --out runs/s0 --seed 0
python -m sentient.average --out runs/avg runs/s0/merged runs/s1/merged runs/s2/merged
python v1/finetune/calibrate.py --model runs/avg --dev $MIX/dev.jsonl
python -m sentient.bench predict --model runs/avg --data $MIX/test.jsonl --out preds/ours.jsonl
python -m sentient.bench predict --openrouter typesafe/jev-1.14-20261014 --data $MIX/test.jsonl --out preds/jev.jsonl
python -m sentient.bench score preds/*.jsonl --data $MIX/test.jsonl --train $MIX/train.jsonl
```
