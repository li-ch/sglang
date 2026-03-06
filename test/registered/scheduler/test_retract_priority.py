import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestRetractPriority(CustomTestCase):
    def test_retract_sorts_low_priority_last(self):
        reqs = [
            SimpleNamespace(priority=10, output_ids=[1], origin_input_ids=[1, 2]),
            SimpleNamespace(priority=5, output_ids=[1], origin_input_ids=[1, 2]),
            SimpleNamespace(priority=0, output_ids=[1], origin_input_ids=[1, 2]),
        ]
        batch = ScheduleBatch(reqs=reqs)
        server_args = SimpleNamespace(
            speculative_algorithm=None,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=False,
        )
        sorted_indices = batch._get_retract_sorted_indices(server_args)
        self.assertEqual(sorted_indices[-1], 2)

    def test_retract_sorts_high_numeric_last_when_low_first(self):
        reqs = [
            SimpleNamespace(priority=0, output_ids=[1], origin_input_ids=[1, 2]),
            SimpleNamespace(priority=5, output_ids=[1], origin_input_ids=[1, 2]),
            SimpleNamespace(priority=10, output_ids=[1], origin_input_ids=[1, 2]),
        ]
        batch = ScheduleBatch(reqs=reqs)
        server_args = SimpleNamespace(
            speculative_algorithm=None,
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
        )
        sorted_indices = batch._get_retract_sorted_indices(server_args)
        self.assertEqual(sorted_indices[-1], 2)


if __name__ == "__main__":
    unittest.main(verbosity=3)
