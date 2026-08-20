import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "windows"
    / "replay_sparse_pq_capture.py"
)
SPEC = importlib.util.spec_from_file_location("replay_sparse_pq_capture", SOURCE)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)


WORKER = "sglang.srt.speculative.eagle_worker_v2.EAGLEWorkerV2"


def _binary_cycle(cycle_id="0", request_id="request", root_q=(0.55, 0.45)):
    states = []
    edges = []
    next_state = 0
    next_edge = 0
    chain = []

    def add_state(depth, path):
        nonlocal next_state, next_edge
        state_id = next_state
        next_state += 1
        states.append(
            {
                "state_id": state_id,
                "depth": depth,
                "p_other_mass": 0.0,
                "q_other_mass": 0.0,
                "target_support_complete": True,
                "draft_support_complete": True,
            }
        )
        if depth == 3:
            return state_id
        p = (0.70, 0.30)
        q = root_q if depth == 0 else (0.55, 0.45)
        for rank, token in enumerate((0, 1)):
            child_id = add_state(depth + 1, path + (token,))
            edge_id = next_edge
            next_edge += 1
            edges.append(
                {
                    "edge_id": edge_id,
                    "parent_id": state_id,
                    "child_id": child_id,
                    "token_id": token,
                    "depth": depth + 1,
                    "branch_rank": rank,
                    "p": p[rank],
                    "q": q[rank],
                    "current_score": q[rank],
                }
            )
            if rank == 0 and all(part == 0 for part in path):
                chain.append(edge_id)
        return state_id

    root = add_state(0, ())
    # Recursive construction appends child edges after deeper descendants.  A
    # membership is ID-based, so sort the all-zero path by depth explicitly.
    all_zero_chain = sorted(
        (
            edge["edge_id"]
            for edge in edges
            if edge["token_id"] == 0
            and all(
                incoming["token_id"] == 0
                for incoming in _ancestry_edges(edge, edges)
            )
        ),
        key=lambda edge_id: next(edge["depth"] for edge in edges if edge["edge_id"] == edge_id),
    )
    return {
        "cycle_id": cycle_id,
        "request_id": request_id,
        "root_state_id": root,
        "max_depth": 3,
        "states": states,
        "edges": edges,
        "topology_memberships": {"current": all_zero_chain},
    }


def _ancestry_edges(edge, edges):
    by_child = {candidate["child_id"]: candidate for candidate in edges}
    result = []
    parent = edge["parent_id"]
    while parent in by_child:
        incoming = by_child[parent]
        result.append(incoming)
        parent = incoming["parent_id"]
    return result


def _cost(cost_id, depth, samples, logical_width=None):
    return {
        "cost_id": cost_id,
        "topology_family": "captured-test",
        "logical_width": logical_width or depth + 1,
        "executed_graph_width": logical_width or depth + 1,
        "max_depth": depth,
        "scope": "full_cycle",
        "samples_ms": samples,
        "active_worker": WORKER,
        "torch_compile_mode": "default",
    }


def _document(cycles=None):
    return {
        "schema_version": 2,
        "artifact_type": "sparse_pq_capture",
        "vocab_size": 2,
        "runtime": {
            "requested_worker": WORKER,
            "active_worker": WORKER,
            "torch_compile_enabled": True,
            "torch_compile_mode": "default",
            "git_head": "0123456789abcdef",
        },
        "sampling": {
            "penalties": {
                "presence": 1.5,
                "frequency": 0.0,
                "repetition": 1.0,
            },
            "transform_order": [
                "presence_frequency",
                "repetition",
                "temperature",
                "top_k",
                "top_p",
            ],
        },
        "cycles": cycles or [_binary_cycle()],
        "width_costs": [
            _cost("depth2", 2, [14.5, 15.0], 3),
            _cost("depth3", 3, [20.771], 4),
        ],
        "policies": [
            {
                "name": "current deterministic tree",
                "kind": "current_deterministic",
                "membership": "current",
                "cost_id": "depth3",
                "measured_frontier_baseline": True,
            },
            {
                "name": "aligned deterministic tree",
                "kind": "aligned_deterministic",
                "logical_width": 4,
                "cost_id": "depth3",
            },
            {
                "name": "irregular variable-fanout tree",
                "kind": "irregular_variable_fanout",
                "logical_width": 4,
                "max_fanout": 2,
                "confidence_power": 1.0,
                "cost_id": "depth3",
            },
            {
                "name": "scalar depth calibration",
                "kind": "scalar_depth_calibration",
                "logical_width": 4,
                "temperature": 0.9,
                "cost_id": "depth3",
            },
            {
                "name": "learned depth calibration",
                "kind": "learned_depth_calibration",
                "logical_width": 4,
                "temperature_by_depth": {"1": 0.9, "2": 1.0, "3": 1.1},
                "cost_id": "depth3",
            },
            {
                "name": "SWOR",
                "kind": "swor",
                "parents": [-1, 0, 1, 2],
                "cost_id": "depth3",
            },
            {
                "name": "confidence-gated two/three-step chain",
                "kind": "confidence_gated_chain",
                "short_depth": 2,
                "long_depth": 3,
                "threshold": 0.6,
                "cost_by_depth": {"2": "depth2", "3": "depth3"},
            },
            {
                "name": "target-aware upper-bound oracle",
                "kind": "target_aware_upper_bound",
                "logical_width": 4,
                "cost_id": "depth3",
                "oracle_only": True,
            },
        ],
        "gate": {"target_tps": 200.0, "funding_tps": 215.0},
    }


class TestReplaySparsePQCapture(unittest.TestCase):
    def test_every_requested_policy_replays_the_same_cycle(self):
        corpus = replay.parse_corpus(_document())

        report = replay.replay_corpus(corpus, "corpus-hash")

        self.assertEqual(len(report["policies"]), 8)
        self.assertEqual(
            {policy["name"] for policy in report["policies"]},
            {
                "current deterministic tree",
                "aligned deterministic tree",
                "irregular variable-fanout tree",
                "scalar depth calibration",
                "learned depth calibration",
                "SWOR",
                "confidence-gated two/three-step chain",
                "target-aware upper-bound oracle",
            },
        )
        for policy in report["policies"]:
            self.assertEqual(policy["cycles"][0]["cycle_id"], "0")
            self.assertGreater(
                policy["aggregate"]["sum_expected_emitted_tokens"], 1.0
            )

    def test_current_membership_expected_length_is_exact(self):
        corpus = replay.parse_corpus(_document())
        report = replay.replay_corpus(corpus)
        current = next(
            policy
            for policy in report["policies"]
            if policy["kind"] == "current_deterministic"
        )

        self.assertAlmostEqual(
            current["aggregate"]["sum_expected_emitted_tokens"],
            1.0 + 0.7 + 0.7**2 + 0.7**3,
            places=12,
        )

    def test_dynamic_depth_uses_total_tokens_over_total_time(self):
        cycles = [
            _binary_cycle("short", root_q=(0.55, 0.45)),
            _binary_cycle("long", root_q=(0.80, 0.20)),
        ]
        corpus = replay.parse_corpus(_document(cycles))
        report = replay.replay_corpus(corpus)
        policy = next(
            item
            for item in report["policies"]
            if item["kind"] == "confidence_gated_chain"
        )

        expected = sum(
            cycle["expected_emitted_tokens"] for cycle in policy["cycles"]
        )
        expected_ms = statistics_mean([14.5, 15.0]) + 20.771
        self.assertAlmostEqual(
            policy["aggregate"]["projected_tps"]["point"],
            1000.0 * expected / expected_ms,
            places=12,
        )

    def test_depth_three_hard_ceiling_rejects_current_cycle_cost(self):
        report = replay.replay_corpus(replay.parse_corpus(_document()))
        ceiling = next(
            item for item in report["hard_ceilings"] if item["cost_id"] == "depth3"
        )

        self.assertAlmostEqual(ceiling["point_tps"], 4000.0 / 20.771, places=9)
        self.assertAlmostEqual(
            ceiling["cost_required_below_ms_for_target"], 20.0
        )
        self.assertAlmostEqual(
            ceiling["cost_required_at_or_below_ms_for_funding"],
            4000.0 / 215.0,
        )
        self.assertTrue(report["gate"]["reject_family"])
        self.assertFalse(report["gate"]["fund_production_implementation"])

    def test_gate_boundaries_are_closed_on_200_and_open_on_215(self):
        document = _single_path_boundary_document()
        report = replay.replay_corpus(replay.parse_corpus(document))

        self.assertAlmostEqual(
            report["gate"]["max_target_aware_oracle_upper_tps"], 200.0
        )
        self.assertAlmostEqual(
            report["gate"]["max_implementable_lower_tps"], 215.0
        )
        self.assertTrue(report["gate"]["reject_family"])
        self.assertTrue(report["gate"]["fund_production_implementation"])

    def test_runtime_cost_provenance_mismatch_is_rejected(self):
        document = _document()
        document["width_costs"][0]["torch_compile_mode"] = "disabled"

        with self.assertRaisesRegex(replay.CaptureInputError, "provenance"):
            replay.parse_corpus(document)

    def test_incomplete_lattice_marks_swor_unavailable_without_aborting_replay(self):
        document = _document()
        document["cycles"][0]["states"][0]["draft_support_complete"] = False

        report = replay.replay_corpus(replay.parse_corpus(document))

        current = next(
            policy
            for policy in report["policies"]
            if policy["kind"] == "current_deterministic"
        )
        swor = next(policy for policy in report["policies"] if policy["kind"] == "swor")
        aligned = next(
            policy
            for policy in report["policies"]
            if policy["kind"] == "aligned_deterministic"
        )
        oracle = next(
            policy
            for policy in report["policies"]
            if policy["kind"] == "target_aware_upper_bound"
        )
        self.assertEqual(current["status"], "ready")
        self.assertEqual(swor["status"], "unavailable")
        self.assertEqual(aligned["status"], "unavailable")
        self.assertEqual(oracle["status"], "unavailable")
        self.assertIn("complete proposal-lattice", swor["reason"])
        self.assertIsNone(report["gate"]["reject_family"])
        self.assertIsNone(report["gate"]["fund_production_implementation"])

    def test_candidate_must_clear_measured_frontier_conservatively(self):
        document = _single_path_boundary_document()
        document["width_costs"] = [
            _cost("baseline", 3, [4000.0 / 220.0], 4),
            _cost("candidate", 3, [4000.0 / 215.0], 4),
            _cost("oracle", 3, [20.0], 4),
        ]
        document["policies"][0]["cost_id"] = "baseline"
        document["policies"][1]["cost_id"] = "candidate"
        document["policies"][2]["cost_id"] = "oracle"

        report = replay.replay_corpus(replay.parse_corpus(document))
        candidate = next(
            policy
            for policy in report["policies"]
            if policy["name"] == "implementable"
        )

        self.assertAlmostEqual(
            report["measured_geometry_frontier"]["projected_tps"]["upper"],
            220.0,
        )
        self.assertFalse(candidate["measured_frontier_comparison"]["clears"])
        self.assertAlmostEqual(
            candidate["measured_frontier_comparison"]["headroom_tps"], -5.0
        )
        self.assertIsNone(
            report["gate"]["max_frontier_clearing_implementable_lower_tps"]
        )
        self.assertIsNone(report["gate"]["fund_production_implementation"])

    def test_missing_measured_frontier_fails_funding_closed(self):
        document = _single_path_boundary_document()
        document["policies"][0].pop("measured_frontier_baseline")

        report = replay.replay_corpus(replay.parse_corpus(document))

        self.assertFalse(report["measured_geometry_frontier"]["available"])
        self.assertIsNone(report["gate"]["fund_production_implementation"])

    def test_membership_must_be_prefix_closed(self):
        document = _document()
        cycle = document["cycles"][0]
        deepest = max(cycle["edges"], key=lambda edge: edge["depth"])["edge_id"]
        cycle["topology_memberships"]["current"] = [deepest]

        with self.assertRaisesRegex(replay.CaptureInputError, "prefix-closed"):
            replay.parse_corpus(document)

    def test_replay_is_byte_deterministic(self):
        corpus = replay.parse_corpus(_document())
        first = json.dumps(replay.replay_corpus(corpus), sort_keys=True)
        second = json.dumps(replay.replay_corpus(corpus), sort_keys=True)

        self.assertEqual(first, second)


def statistics_mean(values):
    return sum(values) / len(values)


def _single_path_boundary_document():
    states = [
        {
            "state_id": index,
            "depth": index,
            "p_other_mass": 0.0,
            "q_other_mass": 0.0,
            "target_support_complete": True,
            "draft_support_complete": True,
        }
        for index in range(4)
    ]
    edges = [
        {
            "edge_id": index,
            "parent_id": index,
            "child_id": index + 1,
            "token_id": 0,
            "depth": index + 1,
            "branch_rank": 0,
            "p": 1.0,
            "q": 1.0,
        }
        for index in range(3)
    ]
    document = _document(
        [
            {
                "cycle_id": "boundary",
                "request_id": "boundary",
                "root_state_id": 0,
                "max_depth": 3,
                "states": states,
                "edges": edges,
                "topology_memberships": {"current": [0, 1, 2]},
            }
        ]
    )
    document["vocab_size"] = 1
    document["width_costs"] = [
        _cost("baseline", 3, [40.0], 4),
        _cost("fund", 3, [4000.0 / 215.0], 4),
        _cost("reject", 3, [20.0], 4),
    ]
    document["policies"] = [
        {
            "name": "measured baseline",
            "kind": "current_deterministic",
            "membership": "current",
            "cost_id": "baseline",
            "measured_frontier_baseline": True,
        },
        {
            "name": "implementable",
            "kind": "aligned_deterministic",
            "logical_width": 4,
            "cost_id": "fund",
        },
        {
            "name": "oracle",
            "kind": "target_aware_upper_bound",
            "logical_width": 4,
            "cost_id": "reject",
            "oracle_only": True,
        },
    ]
    return document


if __name__ == "__main__":
    unittest.main()
