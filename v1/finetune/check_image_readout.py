"""Check that fine-tuning reads a record with images exactly as it is served: the letter logits of every question
through the fine-tuning path (finetune_lora.image_hidden), through StartLuxDecision (v1 engine) and through the v1.1
transformers engine, before any training.

    python finetune/check_image_readout.py --model /path/to/StartLux-Decision-v1.1-4B-AutoThink --data records.jsonl
"""
import argparse
import asyncio
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_lora import ROOT, image_hidden, load_items  # noqa: E402
from startlux_decision import StartLuxDecision  # noqa: E402
from startlux_decision import jevfmt as J  # noqa: E402
from startlux_decision.model import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-pixels", type=int, default=1 << 20)
    a = ap.parse_args()
    from transformers import AutoProcessor, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.model, "decision_config.json")))
    tok = AutoTokenizer.from_pretrained(a.model)
    items, _ = load_items(a.data, tok, 1 << 18, a.max_pixels)
    model, _ = load_model(a.model, device)
    model.eval()
    processor = AutoProcessor.from_pretrained(a.model)
    processor.tokenizer.padding_side = "right"
    head = model.get_output_embeddings().weight
    rows = head.index_select(0, torch.tensor(cfg["letter_token_ids"], device=head.device)).float()
    with torch.no_grad():
        single = [(image_hidden(model, processor, items, [i], a.max_pixels, device) @ rows[:items[i][1]].T)[0]
                  for i in range(len(items))]
        batched = image_hidden(model, processor, items, list(range(len(items))), a.max_pixels, device)
    batched = [(h @ rows[:it[1]].T) for h, it in zip(batched, items)]
    del model
    torch.cuda.empty_cache()

    m = StartLuxDecision(a.model, device=str(device), max_pixels=a.max_pixels, graphs=False)
    base = os.path.dirname(os.path.abspath(a.data))
    served = []
    for line in open(a.data, encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        images = [os.path.join(base, v) if os.path.isfile(os.path.join(base, v)) else v for v in rec["images"]]
        m._media = m._encode_images(images)
        state = m._media_state(rec["state"], len(images))
        qrows = [J.from_systemone(state, q, qid=k) for k, q in rec["questions"].items() if k in (rec.get("targets") or {})]
        served += [z.float().cpu() for z in m._logits(qrows)[0]]
        m._kept = m._media = None
    del m
    torch.cuda.empty_cache()

    sys.path.insert(0, ROOT)
    from startlux_decide.backends.transformers import TransformersBackend
    from startlux_decide.images import load_media

    engine = TransformersBackend(a.model, tok, bidirectional=False, device=str(device))
    engine._load()
    v11 = []
    for prompt_images, it in zip([it[4] for it in items], items):
        prompt, values = prompt_images
        media = load_media(values, max_pixels=a.max_pixels)
        v11.append(torch.tensor(engine._readout(prompt, cfg["letter_token_ids"][:it[1]], media)))

    worst = {"train_vs_batched": 0.0, "train_vs_v1": 0.0, "train_vs_v11": 0.0}
    for s, b, v, w in zip(single, batched, served, v11):
        s, b = s.detach(), b.detach()
        p = torch.softmax(s.cpu(), -1)
        worst["train_vs_batched"] = max(worst["train_vs_batched"], float((p - torch.softmax(b.cpu(), -1)).abs().max()))
        worst["train_vs_v1"] = max(worst["train_vs_v1"], float((p - torch.softmax(v, -1)).abs().max()))
        worst["train_vs_v11"] = max(worst["train_vs_v11"], float((p - torch.softmax(w, -1)).abs().max()))
    print(json.dumps({"questions": len(items), "max_probability_difference": worst}, indent=1))


if __name__ == "__main__":
    main()
