import unittest

from sglang.srt.managers import scheduler as scheduler_mod
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestTauFeedbackLoop(CustomTestCase):
    def _make_scheduler(
        self,
        *,
        current_tau: int,
        min_tau: int,
        max_tau: int,
        target_time_s: float | None,
        sum_s: float = 0.0,
        count: int = 0,
    ):
        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler.current_max_tokens_per_iteration = current_tau
        scheduler.min_tokens_per_iteration = min_tau
        scheduler.max_tokens_per_iteration = max_tau
        scheduler.target_iteration_time_s = target_time_s
        scheduler.iteration_time_sum_s = sum_s
        scheduler.iteration_time_count = count
        scheduler.tau_warmup_iters = 10
        scheduler.tau_target_ema_alpha = 0.1
        return scheduler

    def test_target_time_estimated_after_10_iterations(self):
        scheduler = self._make_scheduler(
            current_tau=100, min_tau=1, max_tau=200, target_time_s=None
        )
        for _ in range(10):
            scheduler._maybe_update_tau(1.0)
        self.assertEqual(scheduler.iteration_time_count, 10)
        self.assertAlmostEqual(scheduler.target_iteration_time_s, 1.0)

    def test_tau_increases_or_decreases_and_clamps(self):
        scheduler = self._make_scheduler(
            current_tau=100, min_tau=50, max_tau=120, target_time_s=1.0
        )
        scheduler._maybe_update_tau(0.5)
        self.assertEqual(scheduler.current_max_tokens_per_iteration, 105)
        scheduler._maybe_update_tau(2.0)
        self.assertEqual(scheduler.current_max_tokens_per_iteration, 99)

        scheduler.current_max_tokens_per_iteration = 120
        scheduler._maybe_update_tau(0.5)
        self.assertEqual(scheduler.current_max_tokens_per_iteration, 120)

        scheduler.current_max_tokens_per_iteration = 50
        scheduler._maybe_update_tau(2.0)
        self.assertEqual(scheduler.current_max_tokens_per_iteration, 50)

    def test_tau_no_change_when_iteration_time_equals_target(self):
        scheduler = self._make_scheduler(
            current_tau=100, min_tau=50, max_tau=150, target_time_s=1.0
        )
        scheduler._maybe_update_tau(1.0)
        self.assertEqual(scheduler.current_max_tokens_per_iteration, 100)

    def test_target_time_updates_with_ema_after_warmup(self):
        scheduler = self._make_scheduler(
            current_tau=100, min_tau=50, max_tau=150, target_time_s=1.0
        )
        scheduler._maybe_update_tau(2.0)
        self.assertAlmostEqual(scheduler.target_iteration_time_s, 1.1)


if __name__ == "__main__":
    unittest.main(verbosity=3)
