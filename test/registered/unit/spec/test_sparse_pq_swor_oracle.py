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


def _logit_row(suffix, target, draft):
    return {
        "suffix": suffix,
        "target_logits": {str(token): logit for token, logit in target.items()},
        "draft_logits": {str(token): logit for token, logit in draft.items()},
    }


def _document(
    *,
    vocab_size,
    parents,
    rows,
    penalties=None,
    root_token=None,
    initial_token_counts=None,
    target_sampling=None,
    draft_sampling=None,
):
    return {
        "schema_version": 1,
        "vocab_size": vocab_size,
        "parents": parents,
        "root_token": vocab_size - 1 if root_token is None else root_token,
        "initial_token_counts": initial_token_counts or {},
        "penalties": penalties or {"presence": 0.0, "frequency": 0.0},
        "target_sampling": target_sampling
        or {"temperature": 1.0, "top_p": 1.0},
        "draft_sampling": draft_sampling
        or {"temperature": 1.0, "top_p": 1.0},
        "rows": rows,
        "cycle_cost_ms": {"target": 15.0, "draft": 5.0},
        "gate": {"target_tps": 200.0, "margin_tps": 10.0},
    }


def _sequential_distribution(
    logits,
    *,
    initial_counts,
    root_token,
    suffix,
    presence,
    frequency,
    repetition,
    temperature,
    top_k,
    top_p,
):
    """Independent scalar model of the active speculative penalty order."""
    counts = dict(initial_counts)
    counts[root_token] = counts.get(root_token, 0) + 1
    for token in suffix:
        counts[token] = counts.get(token, 0) + 1

    transformed = []
    for token, raw_logit in logits.items():
        count = counts.get(token, 0)
        value = raw_logit - frequency * count - (presence if count else 0.0)
        if count:
            value = value * repetition if value < 0.0 else value / repetition
        transformed.append((token, value / temperature))

    transformed.sort(key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        transformed = transformed[:top_k]
    maximum = transformed[0][1]
    weights = [(token, math.exp(value - maximum)) for token, value in transformed]
    total = math.fsum(weight for _, weight in weights)
    probabilities = {token: weight / total for token, weight in weights}

    if top_p < 1.0 and len(probabilities) > 1:
        ascending = sorted(probabilities.items(), key=lambda item: (item[1], item[0]))
        cumulative = 0.0
        cutoff = 0
        while (
            cutoff < len(ascending) - 1
            and cumulative + ascending[cutoff][1] < 1.0 - top_p
        ):
            cumulative += ascending[cutoff][1]
            cutoff += 1
        pivot = ascending[cutoff][1]
        probabilities = {
            token: probability
            for token, probability in probabilities.items()
            if probability >= pivot
        }
        total = math.fsum(probabilities.values())
        probabilities = {
            token: probability / total for token, probability in probabilities.items()
        }
    return probabilities


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

    def test_branch_penalties_match_sequential_target_on_repeated_paths(self):
        target_logits = {
            0: 0.20,
            1: -0.20,
            2: 1.00,
            3: -1.00,
            4: 0.00,
            5: 0.60,
            6: -0.60,
        }
        draft_logits = {
            0: -0.20,
            1: 0.20,
            2: -1.00,
            3: 1.00,
            4: 0.00,
            5: -0.60,
            6: 0.60,
        }
        paths = [
            (),
            (2,),
            (3,),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (3, 3, 3),
        ]
        rows = [
            _logit_row(list(path), target_logits, draft_logits) for path in paths
        ]
        penalties = {"presence": 0.25, "frequency": 0.30, "repetition": 1.50}
        sampling = {"temperature": 0.70, "top_k": 4, "top_p": 0.72}
        document = _document(
            vocab_size=7,
            parents=[-1, 0],
            rows=rows,
            penalties=penalties,
            root_token=4,
            initial_token_counts={"0": 1, "1": 2, "4": 1},
            target_sampling=sampling,
            draft_sampling=sampling,
        )
        evaluator = oracle.SparsePQSworOracle(oracle.parse_document(document))

        # Deliberately traverse branches in an order unrelated to ancestry.
        # Cached state must remain keyed by the immutable suffix.
        for path in (paths[7], paths[3], paths[6], paths[1], paths[5], paths[0], paths[4], paths[2]):
            target, draft = evaluator.distributions_for_suffix(path)
            common = dict(
                initial_counts={0: 1, 1: 2, 4: 1},
                root_token=4,
                suffix=path,
                presence=0.25,
                frequency=0.30,
                repetition=1.50,
                temperature=0.70,
                top_k=4,
                top_p=0.72,
            )
            expected_target = _sequential_distribution(target_logits, **common)
            expected_draft = _sequential_distribution(draft_logits, **common)
            self.assertEqual(set(target), set(expected_target), path)
            self.assertEqual(set(draft), set(expected_draft), path)
            for token, expected in expected_target.items():
                self.assertAlmostEqual(target[token], expected, places=12, msg=(path, token))
            for token, expected in expected_draft.items():
                self.assertAlmostEqual(draft[token], expected, places=12, msg=(path, token))

        # Frequency grows on every repeated token, while repetition is applied
        # once based on presence.  This catches R**count and sibling leakage.
        _, once = evaluator.distributions_for_suffix((3,))
        _, thrice = evaluator.distributions_for_suffix((3, 3, 3))
        self.assertIn(3, once)
        self.assertIn(3, thrice)
        self.assertNotEqual(once[3], thrice[3])

    def test_repetition_penalty_range_matches_sampling_contract(self):
        document = _document(
            vocab_size=3,
            parents=[-1, 0],
            rows=[_row([], {0: 1.0}, {0: 1.0})],
            penalties={"repetition": 0.0},
        )
        with self.assertRaisesRegex(oracle.OracleInputError, "repetition"):
            oracle.parse_document(document)

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
