"""ABC transition interpretation and prevention of invented reward labels."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

from startlux_decide.reward_gate.abc import (
    AnnotationSchemaError, MAX_BYTES, Transition, inspect_abc,
    intervals_from_transitions, mine_candidates, parse_annotation,
)


SOURCE = "data/train/hang_socks/episode_example/annotation.mcap"


class IntervalTests(unittest.TestCase):
    def test_consecutive_duplicates_are_not_retry_and_final_end_is_unknown(self):
        rows = intervals_from_transitions([
            Transition(10, "Pick up sock."), Transition(20, "pick up sock"),
            Transition(30, "Clamp sock"),
        ])
        self.assertEqual([(row.start_ns, row.end_ns) for row in rows], [(10, 30), (30, None)])
        candidates, controls = mine_candidates(rows, SOURCE)
        self.assertEqual(candidates, [])
        self.assertEqual(len(controls), 1)

    def test_repeated_routine_steps_and_keywords_never_become_recovery_labels(self):
        rows = intervals_from_transitions([
            Transition(10, "Pick up sock"), Transition(20, "Clamp sock"),
            Transition(30, "Pick up sock"), Transition(40, "Reset gripper"),
            Transition(50, "Move hanger"),
        ])
        candidates, controls = mine_candidates(rows, SOURCE)
        self.assertEqual(len(candidates), 3)
        repeated = [row for row in candidates if "repeated_action" in row["reasons"]]
        self.assertEqual([row["previous_same_action_index"] for row in repeated], [None, 0])
        for row in candidates + controls:
            self.assertEqual(row["verification_status"], "unverified")
            self.assertFalse(row["training_eligible"])
            self.assertIsNone(row["human_label"])
            self.assertEqual(row["source_path"], SOURCE)
            self.assertNotIn("reward", row)
        self.assertEqual(candidates[-1]["matched_keywords"], ["reset"])

    def test_simultaneous_different_labels_are_refused_not_given_zero_length(self):
        with self.assertRaises(AnnotationSchemaError):
            intervals_from_transitions([Transition(10, "Left grasp"), Transition(10, "Right place")])

    def test_heldout_and_video_paths_are_refused(self):
        rows = intervals_from_transitions([Transition(10, "Reset")])
        for path in [SOURCE.replace("/train/", "/val/"), SOURCE.replace("/train/", "/eval/"),
                     SOURCE.replace("annotation.mcap", "episode.mcap")]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                mine_candidates(rows, path)

    def test_artifact_limits_are_checked_before_network_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "absent"
            for args in [{"max_episodes": 301}, {"max_episodes": 0},
                         {"max_bytes": MAX_BYTES + 1}, {"max_bytes": 0}]:
                with self.subTest(args=args), self.assertRaises(ValueError):
                    inspect_abc(output, **args)
            self.assertFalse(output.exists())


HAS_MCAP = importlib.util.find_spec("mcap_protobuf") is not None


@unittest.skipUnless(HAS_MCAP, "optional mcap-protobuf-support is not installed")
class SchemaTests(unittest.TestCase):
    def _write(self, path, *, schema_name="Annotation", topic="/subtask-annotation", mismatch=False):
        # A real self-describing MCAP, using the observed pinned schema, not a
        # mocked decoder or a stub of a remote service.
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2
        from mcap_protobuf.writer import Writer

        proto = descriptor_pb2.FileDescriptorProto(name="annotation_test.proto", syntax="proto3")
        proto.dependency.append("google/protobuf/timestamp.proto")
        message = proto.message_type.add(name=schema_name)
        message.field.add(name="timestamp", number=1, type=11, type_name=".google.protobuf.Timestamp")
        message.field.add(name="data", number=2, type=9)
        pool = descriptor_pool.DescriptorPool()
        pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
        pool.Add(proto)
        cls = message_factory.GetMessageClass(pool.FindMessageTypeByName(schema_name))
        with path.open("wb") as stream:
            writer = Writer(stream)
            for time, action in [(1_750_000_000_123_456_789, "Grasp sock"),
                                 (1_750_000_002_123_456_789, "Hang sock")]:
                value = cls(data=action)
                value.timestamp.seconds, value.timestamp.nanos = divmod(time, 1_000_000_000)
                writer.write_message(topic, value, log_time=time + int(mismatch), publish_time=time)
            writer.finish()

    def test_actual_protobuf_timestamp_precision_and_schema_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotation.mcap"
            self._write(path)
            intervals, schemas = parse_annotation(path)
        self.assertEqual(intervals[0].start_ns, 1_750_000_000_123_456_789)
        self.assertEqual(intervals[0].end_ns, 1_750_000_002_123_456_789)
        self.assertIsNone(intervals[-1].end_ns)
        self.assertEqual(schemas[0]["name"], "Annotation")
        self.assertEqual(schemas[0]["fields"], ["data", "timestamp"])

    def test_schema_topic_and_time_mismatch_fail_instead_of_guessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "annotation.mcap"
            for args in [{"schema_name": "Other"}, {"topic": "/instruction"}, {"mismatch": True}]:
                with self.subTest(args=args):
                    self._write(path, **args)
                    with self.assertRaises(AnnotationSchemaError):
                        parse_annotation(path)


if __name__ == "__main__":
    unittest.main()
