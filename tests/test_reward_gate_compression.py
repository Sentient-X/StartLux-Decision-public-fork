"""Lossless cache storage must preserve features, identity and physical budgets."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_DEPS = (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("safetensors") is not None)
if HAS_DEPS:
    import torch
    from safetensors.torch import save
    from startlux_decide.reward_gate.data import digest, file_digest
    from startlux_decide.reward_gate.features import FeatureCache


@unittest.skipUnless(HAS_DEPS, "optional torch/safetensors are not installed")
class CompressedFeatureCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        torch.manual_seed(41)
        self.tensors = {
            "visual": torch.randn(2, 1, 4, 768).bfloat16(),
            "times": torch.tensor([0.0, 0.375], dtype=torch.float32),
            "text_mask": torch.tensor([True, False, True]),
            "signed_zero": torch.tensor([0.0, -0.0], dtype=torch.bfloat16),
        }
        self.identity = {"encoder": "local-test", "kind": "visual", "media_sha256": "episode"}

    def test_mixed_bfloat16_float32_boolean_and_signed_zero_roundtrip_bit_exact(self):
        cache = FeatureCache(self.root, reserve_bytes=0, compressed=True)
        key = cache.put(self.identity, self.tensors)
        self.assertTrue(cache.path(key).name.endswith(".safetensors.sz"))
        # Reader's default remains uncompressed; it must autodetect the file.
        reader = FeatureCache(self.root, reserve_bytes=0)
        loaded = reader.read(key)
        self.assertEqual(set(loaded), set(self.tensors))
        for name, expected in self.tensors.items():
            with self.subTest(name=name):
                self.assertEqual(loaded[name].dtype, expected.dtype)
                self.assertEqual(loaded[name].shape, expected.shape)
                self.assertTrue(torch.equal(loaded[name].view(torch.uint8), expected.view(torch.uint8)))
        self.assertEqual(cache.current_bytes, cache.path(key).stat().st_size)
        self.assertEqual(reader.current_bytes, cache.current_bytes)

    def test_exact_safetensors_serialization_with_odd_byte_length_is_reversible(self):
        key = digest(self.identity)
        raw = save(self.tensors, metadata={"key": key, "identity": json.dumps(self.identity)})
        self.assertEqual(len(raw) % 2, 1)
        self.assertEqual(FeatureCache._unpack(FeatureCache._pack(raw)), raw)

    def test_corrupt_payload_digest_and_trailing_data_are_rejected(self):
        cache = FeatureCache(self.root, reserve_bytes=0, compressed=True)
        key = cache.put(self.identity, self.tensors)
        path = cache.path(key)
        original = path.read_bytes()
        for offset in (20, len(original) - 1):
            with self.subTest(offset=offset):
                changed = bytearray(original)
                changed[offset] ^= 1
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    cache.read(key)
        path.write_bytes(original + b"unindexed trailing data")
        with self.assertRaises(ValueError):
            cache.read(key)

    def test_valid_compressed_payload_with_wrong_identity_is_rejected(self):
        cache = FeatureCache(self.root, reserve_bytes=0, compressed=True)
        key = cache.put(self.identity, self.tensors)
        raw = save(self.tensors, metadata={"key": key, "identity": json.dumps({"different": "episode"})})
        cache.path(key).write_bytes(FeatureCache._pack(raw))
        with self.assertRaisesRegex(ValueError, "identity"):
            cache.read(key)

    def test_resuming_with_compression_keeps_existing_raw_cache_bytes_and_hashes(self):
        legacy = FeatureCache(self.root, reserve_bytes=0)
        key = legacy.put(self.identity, self.tensors)
        original_path = legacy.path(key)
        original_hash = file_digest(original_path)
        resumed = FeatureCache(self.root, reserve_bytes=0, compressed=True)
        self.assertEqual(resumed.put(self.identity, self.tensors), key)
        self.assertEqual(resumed.path(key), original_path)
        self.assertEqual(file_digest(original_path), original_hash)
        new_key = resumed.put({**self.identity, "media_sha256": "new-episode"}, self.tensors)
        self.assertTrue(resumed.path(new_key).name.endswith(".safetensors.sz"))
        self.assertEqual(resumed.current_bytes, sum(path.stat().st_size for path in self.root.iterdir()))
        reread = FeatureCache(self.root, reserve_bytes=0)
        self.assertEqual(reread.current_bytes, resumed.current_bytes)
        for entry in (key, new_key):
            self.assertTrue(torch.equal(reread.read(entry)["visual"], self.tensors["visual"]))

    def test_budget_charges_actual_compressed_bytes_and_survives_reopening(self):
        zeros = {"visual": torch.zeros(8, 1, 196, 768, dtype=torch.bfloat16)}
        cache = FeatureCache(self.root, max_bytes=12_000, reserve_bytes=0, compressed=True)
        key = cache.put(self.identity, zeros)
        # A 2.4 MB logical tensor fits a 12 KB physical storage budget.
        self.assertLess(cache.current_bytes, 12_000)
        reopened = FeatureCache(self.root, max_bytes=cache.current_bytes, reserve_bytes=0, compressed=True)
        self.assertEqual(reopened.put(self.identity, zeros), key)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            reopened.put({**self.identity, "media_sha256": "second"}, zeros)
        self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_disk_reserve_is_enforced_before_atomic_write(self):
        cache = FeatureCache(self.root, reserve_bytes=1000, compressed=True)
        with patch("startlux_decide.reward_gate.features.shutil.disk_usage", return_value=SimpleNamespace(free=1000)):
            with self.assertRaisesRegex(ValueError, "disk"):
                cache.put(self.identity, self.tensors)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(cache.current_bytes, 0)
