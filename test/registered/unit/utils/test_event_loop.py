import asyncio
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.utils import event_loop


class TestEventLoopSelection(unittest.TestCase):
    def test_windows_uvicorn_uses_selector_factory(self):
        with (
            patch.object(event_loop, "uvloop", None),
            patch.object(event_loop.sys, "platform", "win32"),
        ):
            self.assertEqual(
                event_loop.uvicorn_loop_name(),
                "sglang.srt.utils.event_loop:windows_selector_loop_factory",
            )

    def test_windows_selector_factory_supports_readers(self):
        loop = event_loop.windows_selector_loop_factory()
        try:
            self.assertTrue(hasattr(loop, "add_reader"))
            self.assertIsInstance(loop, asyncio.SelectorEventLoop)
        finally:
            loop.close()


if __name__ == "__main__":
    unittest.main()
