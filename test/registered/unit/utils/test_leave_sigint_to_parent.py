"""Tests for leave_sigint_to_parent in common.py"""

import subprocess
import sys
import textwrap

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Runs in a fresh interpreter so the handler change cannot leak into the
# test runner. raise_signal drives the same C-level SIGINT action a console
# Ctrl+C reaches, without touching the console the tests run in.
_CHILD = textwrap.dedent(
    """
    import signal

    from sglang.srt.utils import leave_sigint_to_parent

    if {ignore}:
        leave_sigint_to_parent()
    signal.raise_signal(signal.SIGINT)
    print("survived", flush=True)
    """
)


class TestLeaveSigintToParent(CustomTestCase):
    def _run_child(self, ignore: bool) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-X", "utf8", "-c", _CHILD.format(ignore=ignore)],
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_default_handler_raises_keyboard_interrupt(self):
        result = self._run_child(ignore=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KeyboardInterrupt", result.stderr)
        self.assertNotIn("survived", result.stdout)

    def test_worker_survives_sigint(self):
        result = self._run_child(ignore=True)
        if sys.platform == "win32":
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("survived", result.stdout)
        else:
            # Other platforms keep the default handler.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("KeyboardInterrupt", result.stderr)


if __name__ == "__main__":
    import unittest

    unittest.main()
