"""General typed decisions that keep the model's existing skills while it learns robot rewards.

The release models were trained on public datasets reformulated as typed decisions (classification, NLI, QA,
commonsense multiple choice; technical report §2.3.1), and Stage 2 kept part of that mixture to preserve Stage 1.
That mixture is not published, so this builds a stand-in from the training splits of commercially usable public
datasets, written in many formats, never from any benchmark's test split.

    python -m sentient.data.retention --out ~/sd-data/built/retention --per-source 4000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import formats as F

LETTERS = "ABCDEFGHIJ"


def _choice(state, text, options: dict, answer: str, rng: random.Random, keep_order=False):
    return {"type": "choice", "instructions": text, "criteria": F.ordered(options, rng, keep_order)}, {answer: 1.0}


def _state(fields: dict, rng: random.Random):
    if rng.random() < 0.5:
        return fields
    return "\n".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in fields.items())


def snli(row, rng):
    if row["label"] not in (0, 1, 2):
        return None
    names = rng.choice([("entailment", "neutral", "contradiction"), ("follows", "unrelated", "contradicts"),
                        ("yes", "maybe", "no")])
    st = _state({rng.choice(["premise", "text", "statement"]): row["premise"],
                 rng.choice(["hypothesis", "claim", "second_statement"]): row["hypothesis"]}, rng)
    q = rng.choice(["Does the premise support the hypothesis?", "What is the relation between the two statements?",
                    "Given the first statement, is the second true, possibly true, or false?"])
    return st, *_choice(st, q, F.maybe_describe(dict(zip(names, ("must be true", "may or may not be true", "cannot be true"))), rng),
                        names[row["label"]], rng)


def boolq(row, rng):
    st = _state({rng.choice(["passage", "text", "context"]): row["passage"]}, rng)
    spec = {"type": "noul", "instructions": row["question"].strip().rstrip("?") + "?"}
    return st, spec, {"true": 1.0 if row["answer"] else 0.0, "false": 0.0 if row["answer"] else 1.0}


def mc(question, choices: list[str], answer: int, rng, context=None):
    named = rng.random() < 0.5
    ids = choices if named and len(set(choices)) == len(choices) and all(len(c) < 150 for c in choices) else \
        F.letters_or_names(len(choices), rng)
    options = {i: (None if i == c else c) for i, c in zip(ids, choices)}
    fields = {"question": question} if context is None else {"context": context, "question": question}
    st = _state(fields, rng) if rng.random() < 0.6 else (context or "")
    text = question if st == (context or "") else rng.choice(["Which option is correct?", "Choose the best answer.",
                                                               "Pick the right option.", "Answer the question."])
    return st, *_choice(st, text, options, ids[answer], rng)


def commonsense_qa(row, rng):
    labels = row["choices"]["label"]
    if row["answerKey"] not in labels:
        return None
    return mc(row["question"], row["choices"]["text"], labels.index(row["answerKey"]), rng)


def arc(row, rng):
    labels = row["choices"]["label"]
    if row["answerKey"] not in labels:
        return None
    return mc(row["question"], row["choices"]["text"], labels.index(row["answerKey"]), rng)


def hellaswag(row, rng):
    if row["label"] in ("", None):
        return None
    return mc("Which ending is the most plausible continuation?", row["endings"], int(row["label"]), rng, context=row["ctx"])


def openbookqa(row, rng):
    labels = row["choices"]["label"]
    return mc(row["question_stem"], row["choices"]["text"], labels.index(row["answerKey"]), rng)


def winogrande(row, rng):
    if row["answer"] not in ("1", "2"):
        return None
    return mc(row["sentence"].replace("_", "___"), [row["option1"], row["option2"]], int(row["answer"]) - 1, rng,
              context=None)


# (hub id, config, split, builder, licence) — training splits only.
SOURCES = [
    ("stanfordnlp/snli", None, "train", snli, "CC BY-SA 4.0"),
    ("google/boolq", None, "train", boolq, "CC BY-SA 3.0"),
    ("tau/commonsense_qa", None, "train", commonsense_qa, "MIT"),
    ("allenai/ai2_arc", "ARC-Challenge", "train", arc, "CC BY-SA 4.0"),
    ("allenai/ai2_arc", "ARC-Easy", "train", arc, "CC BY-SA 4.0"),
    ("Rowan/hellaswag", None, "train", hellaswag, "MIT"),
    ("allenai/openbookqa", "main", "train", openbookqa, "Apache-2.0"),
    ("allenai/winogrande", "winogrande_xl", "train", winogrande, "CC-BY"),
]


def main():
    from datasets import load_dataset

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=4000)
    ap.add_argument("--dev-share", type=float, default=0.05)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out / "train.jsonl", "w") as tr, open(out / "dev.jsonl", "w") as dv:
        for hub, config, split, build, licence in SOURCES:
            ds = load_dataset(hub, config, split=split)
            idx = list(range(len(ds)))
            random.Random(f"sentient.retention/v1:{hub}:{config}").shuffle(idx)
            n = 0
            for i in idx:
                if n >= a.per_source:
                    break
                rng = random.Random(f"sentient.retention/v1:{hub}:{config}:{i}")
                made = build(ds[i], rng)
                if made is None:
                    continue
                state, spec, target = made
                key = rng.choice(["answer", "decision", "q", "label", "choice", "result"])
                rec = {"state": state, "questions": {key: spec}, "targets": {key: target},
                       "meta": {"source": f"retention:{hub}:{config or ''}", "licence": licence, "kinds": ["retention"],
                                "split": "dev" if rng.random() < a.dev_share else "train"}}
                (dv if rec["meta"]["split"] == "dev" else tr).write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            counts[f"{hub}:{config or ''}"] = n
            print(hub, config, n, flush=True)
    json.dump(counts, open(out / "build.json", "w"), indent=1)


if __name__ == "__main__":
    main()
