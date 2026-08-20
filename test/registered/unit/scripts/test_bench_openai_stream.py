import hashlib
import json
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from scripts.windows import bench_openai_stream

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeResponse:
    def __init__(self, events):
        self._events = events

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self._events)


class TestBenchOpenAIStream(CustomTestCase):
    def test_stream_request_hashes_both_output_channels(self):
        events = [
            (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "reasoning_content": "think",
                                    "content": "answer",
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                + "\n"
            ).encode(),
            (
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {},
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        },
                    }
                )
                + "\n"
            ).encode(),
            b"data: [DONE]\n",
        ]

        with (
            patch.object(
                bench_openai_stream.urllib.request,
                "urlopen",
                return_value=_FakeResponse(events),
            ),
            patch.object(
                bench_openai_stream.time,
                "perf_counter",
                side_effect=[1.0, 2.0, 3.0],
            ),
        ):
            result = bench_openai_stream.stream_request(
                "http://127.0.0.1:30000",
                "model",
                "prompt",
                2,
                10,
                None,
                0.0,
                None,
                None,
                None,
                None,
                None,
                True,
            )

        self.assertEqual(result["reasoning_chars"], 5)
        self.assertEqual(result["content_chars"], 6)
        self.assertEqual(result["nonempty_delta_count"], 1)
        self.assertEqual(result["reasoning_fragment_count"], 1)
        self.assertEqual(result["content_fragment_count"], 1)
        self.assertEqual(result["first_output_delta_chars"], 11)
        self.assertEqual(result["max_output_delta_chars"], 11)
        self.assertEqual(result["trailing_after_last_delta_s"], 1.0)
        self.assertEqual(
            result["output_sha256"],
            hashlib.sha256(b"thinkanswer").hexdigest(),
        )
        self.assertEqual(
            result["reasoning_sha256"],
            hashlib.sha256(b"think").hexdigest(),
        )
        self.assertEqual(
            result["content_sha256"],
            hashlib.sha256(b"answer").hexdigest(),
        )

    def test_validate_result_counts_rejects_invalid_measurements(self):
        valid = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "finish_reason": "length",
        }
        bench_openai_stream.validate_result_counts(
            valid,
            expected_prompt_tokens=10,
            expected_completion_tokens=2,
            label="measurement",
        )

        invalid_cases = [
            {**valid, "prompt_tokens": 9},
            {**valid, "completion_tokens": 1},
            {**valid, "total_tokens": 11},
            {**valid, "finish_reason": "stop"},
        ]
        for result in invalid_cases:
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                bench_openai_stream.validate_result_counts(
                    result,
                    expected_prompt_tokens=10,
                    expected_completion_tokens=2,
                    label="measurement",
                )

    def test_calibrate_prompt_rejects_target_below_template_minimum(self):
        with patch.object(bench_openai_stream, "token_count", return_value=5):
            with self.assertRaisesRegex(ValueError, "below the empty"):
                bench_openai_stream.calibrate_prompt(
                    "http://127.0.0.1:30000",
                    "model",
                    4,
                    10,
                    "sglang",
                )


if __name__ == "__main__":
    unittest.main()
