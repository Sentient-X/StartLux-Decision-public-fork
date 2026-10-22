"""Streaming must preserve content identity and the split quarantine."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from startlux_decide.reward_gate.data import RR_REPO, RR_REVISION, Record, atomic_json, digest
from startlux_decide.reward_gate.streaming import StreamingMedia


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "optional huggingface_hub is not installed")
class StreamingIntegrityTest(unittest.TestCase):
    def test_cached_media_is_verified_and_cross_split_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"a pinned source payload"
            source = root / "source.mp4"
            source.write_bytes(data)
            records = [Record("a", "family-a", "train", "train/a.mp4", "lift", 5),
                       Record("b", "family-b", "dev", "val/b.mp4", "place", 2)]
            files = {r.video: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()} for r in records}

            def write_inventory():
                payload = {"repo": RR_REPO, "revision": RR_REVISION, "files": files}
                atomic_json(root / "inventory.json", {**payload, "digest": digest(payload)})

            write_inventory()
            with self.assertRaisesRegex(ValueError, "cross train/dev"):
                StreamingMedia(records, root / "inventory.json", 1024)
            records = records[:1]
            files.pop("val/b.mp4")
            write_inventory()
            media = StreamingMedia(records, root / "inventory.json", 1024)
            with patch("huggingface_hub.try_to_load_from_cache", return_value=str(source)):
                self.assertEqual(list(media.iter_verified(["train/a.mp4"])), [("train/a.mp4", data)])
                source.write_bytes(b"x" * len(data))
                with self.assertRaisesRegex(ValueError, "content identity"):
                    media.read("train/a.mp4")
            with self.assertRaisesRegex(ValueError, "byte budget"):
                StreamingMedia(records, root / "inventory.json", 1)
