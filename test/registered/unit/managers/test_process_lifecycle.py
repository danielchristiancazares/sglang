import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.srt.managers.io_struct import ShutdownReq
from sglang.srt.managers.tokenizer_manager import (
    ServerStatus,
    TokenizerManager,
)
from sglang.srt.utils import common

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestKillProcessTree(CustomTestCase):
    @patch.object(common, "_wait_for_reap_or_raise")
    @patch.object(common.psutil, "Process")
    def test_waits_for_killed_processes_by_default(self, mock_process, mock_wait):
        parent = MagicMock()
        parent.pid = 987654321
        child = MagicMock()
        child.pid = 987654322
        parent.children.return_value = [child]
        mock_process.return_value = parent

        common.kill_process_tree(parent.pid)

        parent.children.assert_called_once_with(recursive=True)
        child.kill.assert_called_once_with()
        parent.kill.assert_called_once_with()
        mock_wait.assert_called_once_with([child, parent], 60)

    @patch.object(common, "_wait_for_reap_or_raise")
    @patch.object(common.psutil, "Process")
    def test_parent_disappearance_during_child_enumeration_is_safe(
        self, mock_process, mock_wait
    ):
        parent = MagicMock()
        parent.children.side_effect = common.psutil.NoSuchProcess(987654321)
        mock_process.return_value = parent

        common.kill_process_tree(987654321)

        parent.kill.assert_not_called()
        mock_wait.assert_not_called()


class TestRuntimeShutdown(CustomTestCase):
    @patch("sglang.srt.utils.kill_process_tree")
    def test_gc_routed_shutdown_stays_nonblocking(self, mock_kill_process_tree):
        runtime = Runtime.__new__(Runtime)
        runtime.pid = 1234

        runtime.shutdown()

        mock_kill_process_tree.assert_called_once_with(1234, wait_timeout=None)
        self.assertIsNone(runtime.pid)


class TestTokenizerShutdown(CustomTestCase):
    @patch("sglang.srt.managers.tokenizer_manager.get_bool_env_var", return_value=False)
    @patch(
        "sglang.srt.managers.tokenizer_manager.collect_scheduler_processes",
        return_value=[],
    )
    @patch("sglang.srt.managers.tokenizer_manager.sys.exit")
    @patch("sglang.srt.managers.tokenizer_manager.kill_process_tree")
    def test_reaps_children_before_exiting(
        self,
        mock_kill_process_tree,
        mock_exit,
        _mock_collect_scheduler_processes,
        _mock_get_bool_env_var,
    ):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.gracefully_exit = True
        manager.rid_to_state = {}
        manager.server_status = ServerStatus.Up
        manager._subprocess_watchdog = MagicMock()
        manager._dispatch_to_scheduler = MagicMock()
        calls = []
        mock_kill_process_tree.side_effect = lambda *args, **kwargs: calls.append(
            "reaped"
        )
        mock_exit.side_effect = lambda *_args: calls.append("exited")

        asyncio.run(manager.sigterm_watchdog())

        manager._subprocess_watchdog.stop.assert_called_once_with()
        shutdown_req = manager._dispatch_to_scheduler.call_args.args[0]
        self.assertIsInstance(shutdown_req, ShutdownReq)
        mock_kill_process_tree.assert_called_once_with(
            os.getpid(), include_parent=False, wait_timeout=60
        )
        mock_exit.assert_called_once_with(0)
        self.assertEqual(calls, ["reaped", "exited"])


if __name__ == "__main__":
    unittest.main()
