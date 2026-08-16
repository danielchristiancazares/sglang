import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2, EagleDraftWorker


class TestEagleMemoryPoolPreparation(unittest.TestCase):
    def test_outer_worker_delegates_preparation_to_draft(self):
        trace = []
        worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
        worker._draft_worker = SimpleNamespace(
            prepare_memory_pool_allocation=lambda: trace.append("prepare")
        )

        worker.prepare_memory_pool_allocation()

        self.assertEqual(trace, ["prepare"])

    def test_eagle_preparation_is_idempotent(self):
        trace = []
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker._memory_pool_preparation_done = False
        worker.init_token_map = lambda: trace.append("token_map")
        worker.init_lm_head = lambda: trace.append("lm_head")

        worker.prepare_memory_pool_allocation()
        worker.prepare_memory_pool_allocation()

        self.assertEqual(trace, ["token_map", "lm_head"])
        self.assertTrue(worker._memory_pool_preparation_done)

    @staticmethod
    def _scheduler(trace, *, overlap):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            is_startup_weight_load_overlap=overlap
        )
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                memory_pool_config=None,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            ),
            alloc_memory_pool=lambda: trace.append("target_pool"),
            get_memory_pool=lambda: ("request_pool", "token_pool"),
        )
        scheduler.draft_worker = SimpleNamespace(
            prepare_memory_pool_allocation=lambda: trace.append("prepare"),
            alloc_memory_pool=lambda **kwargs: trace.append("draft_pool"),
            init_hicache_draft_plan=lambda: trace.append("hicache"),
        )
        return scheduler

    def test_serial_startup_prepares_draft_before_target_pool(self):
        trace = []
        scheduler = self._scheduler(trace, overlap=False)

        scheduler.init_memory_pools()

        self.assertEqual(
            trace, ["prepare", "target_pool", "draft_pool", "hicache"]
        )

    def test_deferred_weight_loading_preserves_late_preparation(self):
        trace = []
        scheduler = self._scheduler(trace, overlap=True)

        scheduler.init_memory_pools()

        self.assertEqual(trace, ["target_pool", "draft_pool", "hicache"])


if __name__ == "__main__":
    unittest.main()
