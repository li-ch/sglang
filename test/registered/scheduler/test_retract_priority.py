import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_release_req_handles_offload_not_implemented(self):
        req = Mock()
        req.offload_kv_cache.side_effect = NotImplementedError()
        req.reset_for_retract = Mock()
        batch = ScheduleBatch(reqs=[req])
        batch.req_to_token_pool = Mock()
        batch.token_to_kv_pool_allocator = Mock()
        batch.tree_cache = Mock()
        server_args = SimpleNamespace(disaggregation_decode_enable_offload_kvcache=True)

        with patch(
            "sglang.srt.managers.schedule_batch.release_kv_cache"
        ) as release_kv_cache, patch(
            "sglang.srt.managers.schedule_batch.evict_from_tree_cache"
        ) as evict_from_tree_cache:
            batch.release_req(0, 0, server_args)
        release_kv_cache.assert_called_once()
        evict_from_tree_cache.assert_called_once()
        req.reset_for_retract.assert_called_once()

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
