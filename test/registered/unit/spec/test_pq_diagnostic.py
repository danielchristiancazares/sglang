import base64
import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.pq_diagnostic as pq_diagnostic
from sglang.srt.speculative.pq_diagnostic import (
    BranchPQCapture,
    PenaltyConfig,
    SamplingConfig,
    _CaptureWriter,
    branch_counts,
    close_capture_writers,
    make_branch_pq_capture,
    normalize_verifier_topology,
    sparse_distribution_from_logits,
    sparse_distribution_from_probs,
    submit_capture,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPQDiagnostic(CustomTestCase):
    def tearDown(self):
        close_capture_writers()

    def test_topology_normalizes_noncontiguous_branching_order(self):
        # 0 -> [1, 2], 2 -> [3]
        parents, depths, ranks = normalize_verifier_topology(
            [1, -1, 3, -1],
            [-1, 2, -1, -1],
        )

        self.assertEqual(parents, [-1, 0, 0, 2])
        self.assertEqual(depths, [0, 1, 1, 2])
        self.assertEqual(ranks, [-1, 0, 1, 0])

    def test_branch_counts_add_only_true_ancestry_and_never_root_twice(self):
        parents = [-1, 0, 0, 2]
        tokens = [4, 1, 2, 1]
        initial = {4: 2, 1: 1}

        left = branch_counts(initial, parents, tokens, 1)
        right_deep = branch_counts(initial, parents, tokens, 3)
        left_again = branch_counts(initial, parents, tokens, 1)

        self.assertEqual(left, {4: 2, 1: 2})
        self.assertEqual(right_deep, {4: 2, 1: 2, 2: 1})
        self.assertEqual(left_again, left)

    def test_combined_penalties_match_independent_sequential_math(self):
        logits = torch.tensor([0.20, -0.20, 1.00, -1.00, 0.00, 0.60, -0.60])
        counts = {0: 1, 1: 2, 4: 1}
        penalties = PenaltyConfig(presence=0.25, frequency=0.30, repetition=1.50)
        sampling = SamplingConfig(temperature=0.70, top_k=7, top_p=1.0)

        actual = sparse_distribution_from_logits(
            logits, counts, penalties, sampling
        )

        transformed = []
        for token, raw in enumerate(logits.tolist()):
            count = counts.get(token, 0)
            value = raw - 0.30 * count - (0.25 if count else 0.0)
            if count:
                value = value * 1.50 if value < 0.0 else value / 1.50
            transformed.append(value / 0.70)
        maximum = max(transformed)
        weights = [math.exp(value - maximum) for value in transformed]
        total = sum(weights)
        expected = {token: weight / total for token, weight in enumerate(weights)}

        self.assertEqual(set(actual), set(expected))
        for token, probability in expected.items():
            self.assertAlmostEqual(actual[token], probability, places=7)
        # Token zero crosses sign only after additive penalties. This catches a
        # repetition-first implementation.
        self.assertLess(0.20 - 0.30 - 0.25, 0.0)

    def test_repetition_is_presence_scaled_once_not_raised_to_count(self):
        logits = torch.tensor([1.0, 0.0])
        penalties = PenaltyConfig(presence=0.0, frequency=0.0, repetition=2.0)
        sampling = SamplingConfig(temperature=1.0, top_k=2, top_p=1.0)

        once = sparse_distribution_from_logits(
            logits, {0: 1}, penalties, sampling
        )
        thrice = sparse_distribution_from_logits(
            logits, {0: 3}, penalties, sampling
        )

        self.assertEqual(once, thrice)

    def test_greedy_distribution_applies_penalties_then_uses_stable_argmax(self):
        actual = sparse_distribution_from_logits(
            torch.tensor([1.0, 1.0, 0.5]),
            {0: 1},
            PenaltyConfig(presence=0.25, frequency=0.0, repetition=1.0),
            SamplingConfig(temperature=0.0, top_k=3, top_p=1.0),
        )

        self.assertEqual(actual, {1: 1.0})

    def test_exact_q_support_is_normalized_and_rejects_invalid_rows(self):
        actual = sparse_distribution_from_probs(
            torch.tensor([0.0, 0.25, 0.0, 0.75])
        )

        self.assertEqual(actual, {1: 0.25, 3: 0.75})
        with self.assertRaisesRegex(ValueError, "positive mass"):
            sparse_distribution_from_probs(torch.zeros(4))

    def test_cycle_record_contains_branch_exact_p_q_and_ids(self):
        capture = _capture("unused.jsonl")

        record = capture._cycle_record(7)

        cycle = record["cycle"]
        self.assertEqual(record["cycle_ordinal"], 7)
        self.assertEqual(cycle["root_state_id"], 0)
        self.assertEqual(
            [(edge["child_id"], edge["parent_id"], edge["depth"], edge["branch_rank"])
             for edge in cycle["edges"]],
            [(1, 0, 1, 0), (2, 0, 1, 1), (3, 2, 2, 0)],
        )
        states = {state["state_id"]: state for state in cycle["states"]}
        self.assertEqual(states[1]["initial_token_counts"], {"1": 2, "4": 2})
        self.assertEqual(
            states[3]["initial_token_counts"],
            {"1": 2, "2": 1, "4": 2},
        )
        self.assertFalse(states[0]["target_support_complete"])
        self.assertEqual(states[1]["draft_support"], {})
        self.assertIsNone(states[1]["target_argmax_draft_rank"])
        self.assertEqual(record["sampling"]["penalties"]["repetition"], 1.5)
        self.assertEqual(record["realized_accept_length"], 2)
        self.assertEqual(record["scheduler_output_length_snapshot"], 3)
        self.assertEqual(states[0]["target_argmax_draft_rank"], 2)
        hidden = states[0]["draft_hidden"]
        self.assertEqual(hidden["dtype"], "bfloat16")
        self.assertEqual(hidden["shape"], [2])
        self.assertEqual(
            base64.b64decode(hidden["base64"]),
            torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
            .view(torch.uint8)
            .numpy()
            .tobytes(),
        )
        target_hidden = states[0]["target_hidden"]
        self.assertEqual(target_hidden["shape"], [2])
        self.assertEqual(
            base64.b64decode(target_hidden["base64"]),
            torch.tensor([7.0, 8.0], dtype=torch.bfloat16)
            .view(torch.uint8)
            .numpy()
            .tobytes(),
        )

    def test_bounded_writer_appends_complete_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "capture.jsonl")
            capture = _capture(path)

            submit_capture(capture)
            close_capture_writers()

            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["artifact_type"], "sparse_pq_cycle")

    def test_writer_queue_scales_to_capture_limit_with_bounded_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            small = _CaptureWriter(str(Path(directory) / "small.jsonl"), 3)
            large = _CaptureWriter(str(Path(directory) / "large.jsonl"), 256)
            try:
                self.assertEqual(small.queue.maxsize, 3)
                self.assertEqual(large.queue.maxsize, 64)
            finally:
                small.close()
                large.close()

    def test_writer_reserves_capture_limit_before_background_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "limited.jsonl")
            writer = _CaptureWriter(path, 1)
            try:
                writer.submit(_capture(path))
                writer.submit(_capture(path))
            finally:
                writer.close()

            self.assertEqual(writer.submitted, 1)
            self.assertEqual(
                len(Path(path).read_text(encoding="utf-8").splitlines()), 1
            )

    def test_writer_failure_is_logged_and_disables_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory))
            capture = _capture(path)
            submit_capture(capture)
            time.sleep(0.05)
            with self.assertLogs(
                "sglang.srt.speculative.pq_diagnostic", level="ERROR"
            ) as logs:
                submit_capture(capture)
            self.assertTrue(
                any("Disabling failed" in message for message in logs.output)
            )
            submit_capture(capture)
            self.assertNotIn(path, pq_diagnostic._writers)
            self.assertIn(path, pq_diagnostic._disabled_writer_paths)

    def test_writer_construction_failure_is_logged_and_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "capture.jsonl")
            capture = _capture(path)
            with (
                patch.object(
                    threading.Thread,
                    "start",
                    side_effect=RuntimeError("synthetic start failure"),
                ),
                self.assertLogs(
                    "sglang.srt.speculative.pq_diagnostic", level="ERROR"
                ) as logs,
            ):
                submit_capture(capture)
                submit_capture(capture)

            self.assertTrue(
                any("construction failure" in message for message in logs.output)
            )
            self.assertNotIn(path, pq_diagnostic._writers)
            self.assertIn(path, pq_diagnostic._disabled_writer_paths)


    def test_close_does_not_deadlock_after_full_queue_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "failing.jsonl")
            writer = _CaptureWriter(path, 4)
            started = threading.Event()
            release = threading.Event()
            failing = _capture(path)

            def fail_cycle(_ordinal):
                started.set()
                release.wait(timeout=2.0)
                raise RuntimeError("synthetic serialization failure")

            failing._cycle_record = fail_cycle
            writer.submit(failing)
            self.assertTrue(started.wait(timeout=2.0))
            writer.submit(_capture(path))
            writer.submit(_capture(path))
            writer.submit(_capture(path))

            closer = threading.Thread(target=writer.close)
            closer.start()
            release.set()
            closer.join(timeout=2.0)
            self.assertFalse(closer.is_alive())

    def test_concurrent_submissions_reserve_exact_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "concurrent.jsonl")
            writer = _CaptureWriter(path, 3)
            threads = [
                threading.Thread(target=writer.submit, args=(_capture(path),))
                for _ in range(12)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
                self.assertFalse(thread.is_alive())
            writer.close()

            self.assertEqual(writer.submitted, 3)
            self.assertEqual(
                len(Path(path).read_text(encoding="utf-8").splitlines()), 3
            )

    def test_live_capture_factory_snapshots_root_counts(self):
        req = _request()
        batch = SimpleNamespace(reqs=[req])

        capture = make_branch_pq_capture(
            output_path="capture.jsonl",
            max_cycles=3,
            batch=batch,
            raw_target_logits=torch.zeros((4, 5)),
            raw_draft_logits=torch.zeros((3, 5)),
            exact_draft_probs=torch.full((3, 5), 0.2),
            draft_hidden_states=torch.zeros((3, 2), dtype=torch.bfloat16),
            target_hidden_states=torch.zeros((4, 2), dtype=torch.bfloat16),
            node_tokens=torch.tensor([4, 1, 2, 1]),
            retrieve_next_token=torch.tensor([1, -1, 3, -1]),
            retrieve_next_sibling=torch.tensor([-1, 2, -1, -1]),
            draft_sampling_top_k=5,
            active_worker="worker",
            torch_compile_enabled=True,
            torch_compile_mode="default",
            speculative_algorithm="EAGLE",
        )

        self.assertEqual(capture.initial_counts, {4: 2, 1: 1})
        self.assertEqual(capture.penalties.repetition, 1.5)
        self.assertEqual(capture.origin_input_length, 2)
        self.assertEqual(capture.committed_output_length, 3)


def _request():
    params = SimpleNamespace(
        presence_penalty=0.25,
        frequency_penalty=0.30,
        repetition_penalty=1.50,
        temperature=0.70,
        top_k=5,
        top_p=0.90,
        min_p=0.0,
        logit_bias=None,
    )
    return SimpleNamespace(
        rid="request",
        output_ids=[4, 1, 4],
        origin_input_ids=[0, 4],
        sampling_params=params,
        grammar=None,
        custom_logit_processor=None,
    )


def _capture(path):
    return BranchPQCapture(
        output_path=path,
        max_cycles=2,
        request_id="request",
        input_sha256="digest",
        active_worker="worker",
        torch_compile_enabled=True,
        torch_compile_mode="default",
        speculative_algorithm="EAGLE",
        draft_sampling_top_k=5,
        raw_target_logits=torch.tensor(
            [
                [1.0, 0.5, 0.0, -0.5, -1.0],
                [0.8, 0.4, 0.0, -0.4, -0.8],
                [0.6, 0.3, 0.0, -0.3, -0.6],
                [0.4, 0.2, 0.0, -0.2, -0.4],
            ]
        ),
        raw_draft_logits=torch.tensor(
            [
                [0.0, 0.5, 1.0, -0.5, -1.0],
                [0.0, 0.4, 0.8, -0.4, -0.8],
                [0.0, 0.3, 0.6, -0.3, -0.6],
            ]
        ),
        exact_draft_probs=torch.tensor(
            [
                [0.1, 0.2, 0.6, 0.1, 0.0],
                [0.1, 0.2, 0.6, 0.1, 0.0],
                [0.1, 0.2, 0.6, 0.1, 0.0],
            ]
        ),
        draft_hidden_states=torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.bfloat16,
        ),
        target_hidden_states=torch.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
            dtype=torch.bfloat16,
        ),
        accept_lens=torch.tensor([2]),
        origin_input_length=10,
        committed_output_length=3,
        node_tokens=torch.tensor([4, 1, 2, 1]),
        retrieve_next_token=torch.tensor([1, -1, 3, -1]),
        retrieve_next_sibling=torch.tensor([-1, 2, -1, -1]),
        initial_counts={4: 2, 1: 1},
        penalties=PenaltyConfig(0.25, 0.30, 1.50),
        target_sampling=SamplingConfig(0.70, 5, 0.90),
        draft_sampling=SamplingConfig(0.70, 5, 0.90),
        logit_bias={},
    )


if __name__ == "__main__":
    unittest.main()
