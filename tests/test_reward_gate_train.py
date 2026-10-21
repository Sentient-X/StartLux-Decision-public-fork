"""Resume must reproduce optimizer updates, not just reload model weights."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from startlux_decide.reward_gate.data import Record
    from startlux_decide.reward_gate.heads import HeadConfig
    from startlux_decide.reward_gate.train import TrainConfig, train_one
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "reward-gate optional dependencies not installed")
class ResumeTest(unittest.TestCase):
    def test_interrupted_updates_match_uninterrupted_and_reject_stale_inputs(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        records, entries = [], {}
        for i in range(14):
            split = "train" if i < 10 else "dev"
            row = Record(str(i), f"episode/{i}", split, f"{split}/{i}.mp4", "lift object", i % 5 + 1)
            records.append(row)
            entries[row.id] = {"visual": torch.randn(2, 1, 4, 8), "text": torch.randn(3, 8),
                               "text_mask": torch.tensor([True, True, False]),
                               "view_mask": torch.ones(2, 1, dtype=torch.bool), "times": torch.tensor([0., 1.])}
        cfg = TrainConfig(epochs=2, microbatch=2, effective_batch=4)
        head = HeadConfig(feature_dim=8, width=8, num_heads=2, num_layers=1, feedforward_dim=32)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            whole = train_one("late_temporal", records, entries, root / "whole", cfg, "cache-1", head_config=head)
            partial = train_one("late_temporal", records, entries, root / "resumed", cfg, "cache-1",
                                max_updates=2, head_config=head)
            self.assertEqual(partial["status"], "paused_at_update_boundary")
            resumed = train_one("late_temporal", records, entries, root / "resumed", cfg, "cache-1", head_config=head)
            a = torch.load(root / "whole/last.pt", weights_only=True)
            b = torch.load(root / "resumed/last.pt", weights_only=True)
            self.assertEqual(whole["history"], resumed["history"])
            self.assertEqual(a["updates"], 6)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            self.assertEqual(json.loads((root / "whole/predictions.json").read_text()),
                             json.loads((root / "resumed/predictions.json").read_text()))
            with self.assertRaisesRegex(ValueError, "differ"):
                train_one("late_temporal", records, entries, root / "resumed", cfg, "cache-CHANGED", head_config=head)

    def test_held_out_family_never_reaches_optimizer(self):
        records = [Record("a", "shared", "train", "train/a.mp4", "a", 1),
                   Record("b", "shared", "dev", "val/b.mp4", "b", 5)]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "disjoint"):
            train_one("late_temporal", records, {}, Path(tmp), TrainConfig(), "cache")


if __name__ == "__main__":
    unittest.main()
