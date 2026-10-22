import unittest

from startlux_decide.reward_gate.analysis import paired_comparison


class PairedUncertaintyTest(unittest.TestCase):
    def test_pairs_by_identity_and_bootstraps_whole_unequal_families(self):
        candidate = [{"id": str(i), "family": "a" if i < 3 else "b", "gold": 3,
                      "prediction": 3 if i < 3 else 4} for i in range(4)]
        control = [{**r, "prediction": 4 if r["family"] == "a" else 3} for r in reversed(candidate)]
        result = paired_comparison(candidate, control, draws=1000)
        self.assertEqual(result["mae_improvement"], .5)
        self.assertEqual(result["ci95_episode_bootstrap"], [-1, 1])
        self.assertEqual(result["families"], 2)
        with self.assertRaisesRegex(ValueError, "membership"):
            paired_comparison(candidate, control[:-1])
        control[0]["gold"] = 2
        with self.assertRaisesRegex(ValueError, "target"):
            paired_comparison(candidate, control)
