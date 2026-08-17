import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "windows"
    / "sparse_pq_swor_oracle.py"
)
SPEC = importlib.util.spec_from_file_location("sparse_pq_swor_oracle", SOURCE)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def _row(suffix, target, draft):
    return {
        "suffix": suffix,
        "target_logits": {
            str(token): math.log(probability) for token, probability in target.items()
        },
        "draft_logits": {
            str(token): math.log(probability) for token, probability in draft.items()
        },
    }


def _document(*, vocab_size, parents, rows, penalties=None):
    return {
        "schema_version": 1,
        "vocab_size": vocab_size,
        "parents": parents,
        "root_token": vocab_size - 1,
        "initial_token_counts": {},
        "penalties": penalties or {"presence": 0.0, "frequency": 0.0},
        "target_sampling": {"temperature": 1.0, "top_p": 1.0},
        "draft_sampling": {"temperature": 1.0, "top_p": 1.0},
        "rows": rows,
        "cycle_cost_ms": {"target": 15.0, "draft": 5.0},
        "gate": {"target_tps": 200.0, "margin_tps": 10.0},
    }


class TestSparsePQSworOracle(unittest.TestCase):
    def test_exact_two_rank_swor_integrates_proposal_and_rejection(self):
        # First-rank overlap is .7.  Only q(token=1) can reject, with mass .3.
        # Its residual is p={0:1}; the remaining q is {0:.5,2:.5}, so the
        # second rank accepts another .3 * .5 = .15 of cycles.
        document = _document(
            vocab_size=4,
            parents=[-1, 0, 0],
            rows=[
                _row(
                    [],
                    {0: 0.50, 1: 0.30, 2: 0.20},
                    {0: 0.20, 1: 0.60, 2: 0.20},
                )
            ],
        )

        report = oracle.SparsePQSworOracle(oracle.parse_document(document)).evaluate()

        self.assertAlmostEqual(
            report["expected"]["accepted_drafts_per_cycle"], 0.85, places=12
        )
        self.assertAlmostEqual(
            report["expected"]["emitted_tokens_per_cycle"], 1.85, places=12
        )
        by_node = report["expected"]["accepted_probability_by_node"]
        self.assertAlmostEqual(by_node[1], 0.70, places=12)
        self.assertAlmostEqual(by_node[2], 0.15, places=12)

    def test_zero_q_fallback_is_uniform_over_unrejected_vocabulary(self):
        # q's sole token is rejected.  The next proposal is uniform across the
        # other three vocabulary IDs and accepts target token 1 with mass 1/3.
        document = _document(
            vocab_size=4,
            parents=[-1, 0, 0],
            rows=[_row([], {1: 1.0}, {0: 1.0})],
        )

        report = oracle.SparsePQSworOracle(oracle.parse_document(document)).evaluate()

        self.assertAlmostEqual(
            report["expected"]["accepted_drafts_per_cycle"], 1.0 / 3.0, places=12
        )
        by_node = report["expected"]["accepted_probability_by_node"]
        self.assertEqual(by_node[1], 0.0)
        self.assertAlmostEqual(by_node[2], 1.0 / 3.0, places=12)

    def test_presence_penalty_is_rebuilt_for_each_accepted_branch(self):
        log_three = math.log(3.0)
        rows = [
            _row([], {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),
            _row([1], {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),
            _row([2], {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),
        ]
        document = _document(
            vocab_size=4,
            parents=[-1, 0, 1],
            rows=rows,
            penalties={"presence": log_three, "frequency": 0.0},
        )
        evaluator = oracle.SparsePQSworOracle(oracle.parse_document(document))

        target_after_one, draft_after_one = evaluator.distributions_for_suffix((1,))
        target_after_two, draft_after_two = evaluator.distributions_for_suffix((2,))

        for distribution in (target_after_one, draft_after_one):
            self.assertAlmostEqual(distribution[1], 0.25, places=12)
            self.assertAlmostEqual(distribution[2], 0.75, places=12)
        for distribution in (target_after_two, draft_after_two):
            self.assertAlmostEqual(distribution[1], 0.75, places=12)
            self.assertAlmostEqual(distribution[2], 0.25, places=12)

    def test_expectation_recurses_through_token_specific_branch_rows(self):
        document = _document(
            vocab_size=3,
            parents=[-1, 0, 1],
            rows=[
                _row([], {0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}),
                _row([0], {0: 1.0}, {0: 1.0}),
                _row([1], {1: 1.0}, {1: 1.0}),
            ],
        )

        report = oracle.SparsePQSworOracle(oracle.parse_document(document)).evaluate()

        self.assertAlmostEqual(
            report["expected"]["accepted_drafts_per_cycle"], 2.0, places=12
        )
        self.assertEqual(
            report["expected"]["accepted_probability_by_node"],
            [0.0, 1.0, 1.0],
        )

    def test_frequency_penalty_counts_only_tokens_on_that_branch(self):
        document = _document(
            vocab_size=4,
            parents=[-1, 0],
            rows=[
                _row([], {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),
                _row([1, 1], {1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}),
            ],
            penalties={"presence": 0.0, "frequency": math.log(2.0)},
        )
        evaluator = oracle.SparsePQSworOracle(oracle.parse_document(document))

        target, _ = evaluator.distributions_for_suffix((1, 1))

        self.assertAlmostEqual(target[1], 0.2, places=12)
        self.assertAlmostEqual(target[2], 0.8, places=12)

    def test_projection_gate_is_strict_and_uses_all_measured_phase_costs(self):
        gate = oracle.PromotionGate(target_tps=200.0, margin_tps=10.0)

        boundary = oracle.project_throughput(
            4.2, {"target": 15.0, "draft": 3.0, "seam": 2.0}, gate
        )
        above = oracle.project_throughput(
            4.201, {"target": 15.0, "draft": 3.0, "seam": 2.0}, gate
        )

        self.assertAlmostEqual(boundary["total_ms"], 20.0)
        self.assertAlmostEqual(boundary["projected_tps"], 210.0)
        self.assertAlmostEqual(
            boundary["gate"]["required_emitted_tokens_per_cycle_exclusive"],
            4.2,
        )
        self.assertFalse(boundary["gate"]["passes"])
        self.assertTrue(above["gate"]["passes"])

    def test_margin_must_be_explicit_and_positive(self):
        document = _document(
            vocab_size=3,
            parents=[-1, 0],
            rows=[_row([], {0: 1.0}, {0: 1.0})],
        )
        document["gate"]["margin_tps"] = 0.0

        with self.assertRaisesRegex(oracle.OracleInputError, "margin_tps"):
            oracle.parse_document(document)

    def test_cli_can_enforce_a_failed_promotion_gate(self):
        document = _document(
            vocab_size=3,
            parents=[-1, 0],
            rows=[_row([], {0: 1.0}, {0: 1.0})],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = oracle.main(
                    [str(path), "--json", "--require-promotion-gate"]
                )

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
