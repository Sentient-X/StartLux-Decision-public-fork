"""Evidence-integrity regressions without network downloads or pretrained weights."""
from dataclasses import asdict
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from startlux_decide.reward_gate.data import (
    RR_REPO, RR_REVISION, Record, atomic_json, digest, file_digest,
    prepare, read_manifest, validate_records,
)

HAS_HUB = importlib.util.find_spec("huggingface_hub") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_FEATURE_DEPS = HAS_TORCH and importlib.util.find_spec("safetensors") is not None
if HAS_FEATURE_DEPS:
    import torch
    from safetensors.torch import save_file
    from startlux_decide.reward_gate.features import FeatureCache, load_features, sample_video


def source_row(index, task="lift the cube", reward=5):
    return {"file_name": f"bridge/bridge_originalsplit_train_index_{index}.mp4",
            "task": task, "reward": reward}


def write_manifest(path, records):
    value = {"repo": RR_REPO, "revision": RR_REVISION,
             "records": [asdict(row) for row in records]}
    value["digest"] = digest(value)
    atomic_json(path, value)
    return value["digest"]


class MembershipIntegrityTest(unittest.TestCase):
    @unittest.skipUnless(HAS_HUB, "optional huggingface_hub is not installed")
    def test_selection_quarantines_validation_and_test_families_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_rows = [source_row(0), source_row(0, task="place the cube", reward=2),
                          source_row(1), source_row(2)]
            train_rows[-1]["file_name"] = "bridge/bridge_originalsplit_train_index_2_clipped_00.mp4"
            for name, rows in (("train", train_rows), ("val", [source_row(1)])):
                (root / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            info = SimpleNamespace(siblings=[
                SimpleNamespace(rfilename="val/" + source_row(1)["file_name"]),
                SimpleNamespace(rfilename="test/" + source_row(2)["file_name"]),
            ])
            downloaded = []

            def local_metadata(repo, filename, **kwargs):
                self.assertEqual((repo, kwargs["revision"]), (RR_REPO, RR_REVISION))
                downloaded.append(filename)
                return str(root / (filename.split("/")[0] + ".jsonl"))

            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download", side_effect=local_metadata):
                api.return_value.dataset_info.return_value = info
                manifest = prepare(root / "manifest.json", train_count=1, dev_count=1)
            records, _ = read_manifest(root / "manifest.json")
            training = [record for record in records if record.split == "train"]
            self.assertEqual(len(training), 2)  # keep a whole source family
            self.assertEqual({record.family for record in training}, {"bridge/bridge_originalsplit_train_index_0"})
            self.assertEqual(manifest["train_rows_quarantined"], 2)
            self.assertEqual(downloaded, ["train/metadata.jsonl", "val/metadata.jsonl"])

    def test_unknown_lineage_and_source_family_split_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown episode lineage"):
            Record.parse({"file_name": "bridge/untraceable_clip.mp4", "task": "lift", "reward": 4}, "train", "train")
        original = Record.parse(source_row(12), "train", "train")
        clipped = source_row(12)
        clipped["file_name"] = "bridge/bridge_originalsplit_train_index_12_clipped_2.mp4"
        held_out = Record.parse(clipped, "dev", "val")
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_records([original, held_out])
        with self.assertRaisesRegex(ValueError, "test labels"):
            Record.parse(source_row(13), "dev", "test")

    def test_modified_manifest_and_rehashed_wrong_revision_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path, [Record.parse(source_row(0), "train", "train")])
            value = json.loads(path.read_text())
            value["records"][0]["reward"] = 1
            atomic_json(path, value)
            with self.assertRaisesRegex(ValueError, "digest/revision"):
                read_manifest(path)
            value["revision"] = "different-dataset-revision"
            del value["digest"]
            value["digest"] = digest(value)
            atomic_json(path, value)
            with self.assertRaisesRegex(ValueError, "digest/revision"):
                read_manifest(path)


@unittest.skipUnless(HAS_FEATURE_DEPS, "optional torch/safetensors are not installed")
class FeatureIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.json"
        self.records = [Record.parse(source_row(0, task="lift red cube"), "train", "train"),
                        Record.parse(source_row(1, task="lift blue cube"), "dev", "val")]
        membership_digest = write_manifest(self.manifest, self.records)
        self.cache = FeatureCache(self.root / "features", reserve_bytes=0)
        spec = {"test_encoder": "local-test-features"}
        encoder_digest = digest(spec)
        rows = []
        for i, record in enumerate(self.records):
            media_digest = digest(["video", i])
            visual_key = self.cache.put({"encoder": encoder_digest, "kind": "visual", "media_sha256": media_digest},
                                       {"visual": torch.full((8, 1, 196, 768), float(i), dtype=torch.bfloat16),
                                        "times": torch.linspace(0, 1, 8), "seconds": torch.linspace(0, 7, 8),
                                        "view_mask": torch.ones(8, 1, dtype=torch.bool)})
            text_key = self.cache.put({"encoder": encoder_digest, "kind": "text", "task": record.task},
                                     {"text": torch.full((64, 768), float(i), dtype=torch.bfloat16),
                                      "text_mask": torch.ones(64, dtype=torch.bool)})
            rows.append({"record_id": record.id, "visual_key": visual_key, "text_key": text_key,
                         "media_sha256": media_digest, "visual_sha256": file_digest(self.cache.path(visual_key)),
                         "text_sha256": file_digest(self.cache.path(text_key))})
        self.index = {"schema": 1, "manifest_digest": membership_digest, "encoder_digest": encoder_digest,
                      "encoder": spec, "complete": True, "rows": rows}
        atomic_json(self.root / "index.json", self.index)

    def test_corrupted_tensor_payload_is_rejected_even_with_valid_metadata(self):
        records, features, _ = load_features(self.manifest, self.root)
        self.assertEqual(len(records), 2)
        self.assertEqual(features[records[1].id]["visual"][0, 0, 0, 0].item(), 1)
        path = self.cache.path(self.index["rows"][0]["visual_key"])
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "bytes"):
            load_features(self.manifest, self.root)

    def test_valid_feature_files_cannot_be_reassigned_to_other_instructions(self):
        first, second = self.index["rows"]
        first["text_key"], second["text_key"] = second["text_key"], first["text_key"]
        first["text_sha256"], second["text_sha256"] = second["text_sha256"], first["text_sha256"]
        atomic_json(self.root / "index.json", self.index)
        with self.assertRaises(ValueError):
            load_features(self.manifest, self.root)

    def test_visual_cache_identity_must_bind_to_index_media(self):
        first, second = self.index["rows"]
        first["visual_key"], second["visual_key"] = second["visual_key"], first["visual_key"]
        first["visual_sha256"], second["visual_sha256"] = second["visual_sha256"], first["visual_sha256"]
        atomic_json(self.root / "index.json", self.index)
        with self.assertRaises(ValueError):
            load_features(self.manifest, self.root)

    def test_encoder_specification_cannot_change_under_same_digest(self):
        self.index["encoder"]["test_encoder"] = "new-encoder-in-stale-index"
        atomic_json(self.root / "index.json", self.index)
        with self.assertRaises(ValueError):
            load_features(self.manifest, self.root)

    def test_cross_split_exact_media_bytes_are_rejected_despite_different_names(self):
        self.index["rows"][1]["media_sha256"] = self.index["rows"][0]["media_sha256"]
        atomic_json(self.root / "index.json", self.index)
        with self.assertRaises(ValueError):
            load_features(self.manifest, self.root)

    def test_cache_file_cannot_impersonate_another_identity(self):
        key = self.index["rows"][0]["text_key"]
        save_file({"text": torch.zeros(2, 12)}, str(self.cache.path(key)),
                  metadata={"key": key, "identity": json.dumps({"wrong": "identity"})})
        with self.assertRaisesRegex(ValueError, "identity"):
            self.cache.read(key)


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("transformers") is not None,
                     "optional torch/transformers are not installed")
class FrozenEncoderTest(unittest.TestCase):
    def test_local_siglip_towers_are_frozen_and_return_patch_tokens(self):
        from PIL import Image
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, SiglipConfig, SiglipImageProcessor, SiglipModel
        from startlux_decide.reward_gate.features import FrozenSiglip
        backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "lift": 2, "cube": 3}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
        image_processor = SiglipImageProcessor(size={"height": 224, "width": 224})

        class LocalProcessor:
            def __call__(self, **kwargs):
                return image_processor(**kwargs)

        processor = LocalProcessor()
        processor.tokenizer = tokenizer
        processor.image_processor = image_processor
        # No transformer blocks or pretrained weights: only actual SigLIP
        # embeddings/pooling, preserving the production patch geometry.
        config = SiglipConfig(
            text_config={"vocab_size": 4, "hidden_size": 768, "intermediate_size": 16,
                         "num_hidden_layers": 0, "num_attention_heads": 12, "max_position_embeddings": 64},
            vision_config={"hidden_size": 768, "intermediate_size": 16, "num_hidden_layers": 0,
                           "num_attention_heads": 12, "image_size": 224, "patch_size": 16},
        )
        model = SiglipModel(config)
        with patch("transformers.AutoProcessor.from_pretrained", return_value=processor), \
             patch("transformers.AutoModel.from_pretrained", return_value=model):
            encoder = FrozenSiglip(local_files_only=True)
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        model.train()  # extraction must restore evaluation after accidental caller mutation
        visual = encoder.images([Image.new("RGB", (32, 48), (60, 80, 90))])
        text, mask = encoder.text("lift cube")
        self.assertFalse(model.training)
        self.assertEqual(visual.shape, (1, 196, 768))
        self.assertEqual(text.shape, (64, 768))
        self.assertEqual(mask.sum().item(), 2)
        self.assertFalse(visual.requires_grad)
        self.assertFalse(text.requires_grad)
        self.assertIsNone(visual.grad_fn)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        from startlux_decide.reward_gate.heads import HeadConfig, make_head, outcome_loss
        # Clone outside inference_mode, as a safetensors cache read does, then
        # backpropagate through a real head without touching the encoder.
        head = make_head("late_temporal", HeadConfig(width=16, num_heads=4,
                                                     feedforward_dim=32, dropout=0))
        logits = head(visual.clone()[None, :, None], text.clone()[None], mask.clone()[None],
                      torch.ones(1, 1, 1, dtype=torch.bool), torch.zeros(1, 1))
        outcome_loss(logits, torch.tensor([4])).backward()
        self.assertGreater(head.visual_projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "64-token"):
            encoder.text(" ".join(["cube"] * 65))


@unittest.skipUnless(HAS_FEATURE_DEPS and importlib.util.find_spec("av") is not None,
                     "optional PyAV/torch/safetensors are not installed")
class VideoSamplingTest(unittest.TestCase):
    def test_variable_timestamps_and_short_video_repeat_real_frames(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timestamps.mkv"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("ffv1", rate=10)
                stream.width = stream.height = 16
                stream.pix_fmt = "bgr0"
                stream.time_base = Fraction(1, 10)
                for pts, intensity in ((0, 0), (1, 80), (9, 200)):
                    frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), intensity, dtype=np.uint8), format="rgb24")
                    frame.pts = pts
                    frame.time_base = Fraction(1, 10)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            images, times = sample_video(path, frame_count=5)
            torch.testing.assert_close(times, torch.tensor([0.0, 0.1, 0.1, 0.9, 0.9]))
            self.assertEqual([image.getpixel((0, 0))[0] for image in images], [0, 80, 80, 200, 200])
