import unittest
from types import SimpleNamespace

from sglang.srt.managers import scheduler as scheduler_mod
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestAdmissionBackpressure(CustomTestCase):
    def _make_scheduler(self, available_tokens: int, evictable_tokens: int):
        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: available_tokens
        )
        scheduler.tree_cache = SimpleNamespace(
            evictable_size=lambda: evictable_tokens
        )
        return scheduler

    def test_admission_headroom_uses_available_plus_evictable(self):
        scheduler = self._make_scheduler(available_tokens=10, evictable_tokens=6)
        self.assertEqual(scheduler._get_admission_headroom_tokens(), 16)

    def test_admission_headroom_handles_allocator_failures(self):
        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        scheduler.tree_cache = SimpleNamespace(
            evictable_size=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        self.assertEqual(scheduler._get_admission_headroom_tokens(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
