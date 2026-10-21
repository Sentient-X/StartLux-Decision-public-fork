"""Behavioral controls for the frozen-feature reward experiment."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from startlux_decide.reward_gate.heads import (
        HEAD_KINDS, HeadConfig, make_head, outcome_loss, ranked_probability_loss,
    )


@unittest.skipUnless(HAS_TORCH, "optional torch is not installed")
class RewardGateHeadsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        self.config = HeadConfig(feature_dim=12, width=16, num_heads=4,
                                 feedforward_dim=32, dropout=0)
        self.visual = torch.randn(2, 3, 2, 4, 12)
        self.text = torch.randn(2, 4, 12)
        self.text_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
        self.view_mask = torch.tensor([[[True, False], [True, True], [True, False]],
                                       [[True, True], [False, True], [True, True]]])
        self.times = torch.tensor([[0.0, 0.4, 1.0], [0.0, 0.5, 1.0]])

    def forward(self, head):
        return head(self.visual, self.text, self.text_mask, self.view_mask, self.times)

    def test_finite_logits_and_padding_invariance(self):
        for kind in HEAD_KINDS:
            with self.subTest(kind=kind):
                head = make_head(kind, self.config).eval()
                expected = self.forward(head)
                self.assertEqual(expected.shape, (2, 5))
                self.assertTrue(torch.isfinite(expected).all())
                visual = self.visual.masked_fill(~self.view_mask[..., None, None], float("nan"))
                text = self.text.masked_fill(~self.text_mask[..., None], float("nan"))
                # Append an entirely absent camera and padded language tokens.
                visual = torch.cat((visual, torch.full((2, 3, 1, 4, 12), float("nan"))), dim=2)
                views = torch.cat((self.view_mask, torch.zeros(2, 3, 1, dtype=torch.bool)), dim=2)
                text = torch.cat((text, torch.full((2, 2, 12), float("nan"))), dim=1)
                text_mask = torch.cat((self.text_mask, torch.zeros(2, 2, dtype=torch.bool)), dim=1)
                actual = head(visual, text, text_mask, views, self.times)
                torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)

    def test_orderless_is_invariant_to_frame_permutation_and_times(self):
        head = make_head("late_orderless", self.config).eval()
        expected = self.forward(head)
        permutation = torch.tensor([2, 0, 1])
        actual = head(self.visual[:, permutation], self.text, self.text_mask,
                      self.view_mask[:, permutation], torch.zeros_like(self.times))
        torch.testing.assert_close(expected, actual, atol=2e-6, rtol=2e-6)

    def test_temporal_controls_use_temporal_association(self):
        for kind in ("late_temporal", "pooled_temporal"):
            with self.subTest(kind=kind):
                head = make_head(kind, self.config).eval()
                expected = self.forward(head)
                reversed_evidence = head(self.visual.flip(1), self.text, self.text_mask,
                                         self.view_mask.flip(1), self.times)
                self.assertGreater((expected - reversed_evidence).abs().max().item(), 1e-5)
                # Joint permutation of observations/timestamps should retain identity.
                same_evidence = head(self.visual.flip(1), self.text, self.text_mask,
                                     self.view_mask.flip(1), self.times.flip(1))
                torch.testing.assert_close(expected, same_evidence, atol=2e-6, rtol=2e-6)

    def test_pooled_control_discards_patch_arrangement_but_candidate_keeps_it(self):
        permutation = torch.tensor([3, 2, 1, 0])
        for kind in ("pooled_temporal", "late_temporal"):
            with self.subTest(kind=kind):
                head = make_head(kind, self.config).eval()
                original = self.forward(head)
                shuffled = head(self.visual[:, :, :, permutation], self.text,
                                self.text_mask, self.view_mask, self.times)
                if kind == "pooled_temporal":
                    torch.testing.assert_close(original, shuffled, atol=2e-6, rtol=2e-6)
                else:
                    self.assertGreater((original - shuffled).abs().max().item(), 1e-5)

    def test_real_features_and_projections_receive_gradients(self):
        for kind in HEAD_KINDS:
            with self.subTest(kind=kind):
                head = make_head(kind, self.config)
                visual = self.visual.clone().requires_grad_()
                text = self.text.clone().requires_grad_()
                logits = head(visual, text, self.text_mask, self.view_mask, self.times)
                loss = outcome_loss(logits, torch.tensor([0, 4]))
                loss.backward()
                self.assertGreater(visual.grad.abs().sum().item(), 0)
                self.assertGreater(text.grad.abs().sum().item(), 0)
                self.assertEqual(visual.grad.masked_select(~self.view_mask[..., None, None]).abs().sum().item(), 0)
                self.assertEqual(text.grad.masked_select(~self.text_mask[..., None]).abs().sum().item(), 0)
                for layer in (head.visual_projection, head.text_projection):
                    self.assertTrue(torch.isfinite(layer.weight.grad).all())
                    self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
                for parameter in head.parameters():
                    self.assertIsNotNone(parameter.grad)

    def test_default_heads_are_small_and_parameter_matched(self):
        counts = [sum(parameter.numel() for parameter in make_head(kind).parameters())
                  for kind in HEAD_KINDS]
        self.assertLess(max(counts), 1_000_000)
        self.assertLess(max(counts) / min(counts), 1.1)

    def test_bfloat16_cache_features_work_with_float32_head(self):
        head = make_head("late_temporal", self.config)
        actual = head(self.visual.bfloat16(), self.text.bfloat16(), self.text_mask,
                      self.view_mask, self.times)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.isfinite(actual).all())

    def test_empty_evidence_and_invalid_times_are_rejected(self):
        head = make_head("late_temporal", self.config)
        with self.assertRaises(ValueError):
            head(self.visual, self.text, torch.zeros_like(self.text_mask), self.view_mask, self.times)
        views = self.view_mask.clone()
        views[0, 1] = False
        with self.assertRaises(ValueError):
            head(self.visual, self.text, self.text_mask, views, self.times)
        with self.assertRaises(ValueError):
            head(self.visual, self.text, self.text_mask, self.view_mask, self.times + 2)

    def test_ordinal_loss_penalizes_distant_errors(self):
        target = torch.tensor([0])
        nearby = torch.tensor([[-10.0, 10.0, -10.0, -10.0, -10.0]])
        distant = torch.tensor([[-10.0, -10.0, -10.0, -10.0, 10.0]])
        self.assertGreater(ranked_probability_loss(distant, target).item(),
                           ranked_probability_loss(nearby, target).item())
        self.assertGreater(outcome_loss(distant, target).item(), outcome_loss(nearby, target).item())


if __name__ == "__main__":
    unittest.main()
