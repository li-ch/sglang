import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_mod
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class _FakeReq:
    lora_id = None

    def init_next_round_input(self, tree_cache):
        return None


class _FakePrefillAdder:
    last_rem_input_tokens = None

    def __init__(
        self,
        page_size,
        tree_cache,
        token_to_kv_pool_allocator,
        running_batch,
        new_token_ratio,
        rem_input_tokens,
        rem_chunk_tokens,
        mixed_with_decode_tokens=0,
        priority_scheduling_preemption_threshold=0,
        max_prefill_bs=0,
        max_running_requests=None,
        prefill_max_requests=None,
        prefill_delayer_single_pass=None,
        dllm_config=None,
    ):
        type(self).last_rem_input_tokens = rem_input_tokens
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        return AddReqResult.NO_TOKEN


class TestTauPrefillBudget(CustomTestCase):
    def _make_scheduler(self, *, running_bs, max_prefill_tokens, max_tokens_per_iteration):
        scheduler = scheduler_mod.Scheduler.__new__(scheduler_mod.Scheduler)
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.enable_priority_preemption = False
        scheduler.running_batch = SimpleNamespace(
            batch_is_full=False,
            reqs=[MagicMock() for _ in range(running_bs)],
        )
        scheduler.waiting_queue = [_FakeReq()]
        scheduler.chunked_req = None
        scheduler.dllm_config = None
        scheduler.enable_hierarchical_cache = False
        scheduler.policy = SimpleNamespace(
            calc_priority=lambda waiting_queue, running_batch: False
        )
        scheduler.chunked_prefill_size = None
        scheduler.enable_dynamic_chunking = False
        scheduler.page_size = 1
        scheduler.tree_cache = MagicMock()
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.new_token_ratio = 1.0
        scheduler.max_prefill_tokens = max_prefill_tokens
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 0
        scheduler.max_running_requests = None
        scheduler.server_args = SimpleNamespace(
            prefill_max_requests=None,
            max_tokens_per_iteration=max_tokens_per_iteration,
        )
        scheduler.enable_lora = False
        scheduler.enable_lora_overlap_loading = False
        scheduler.lora_overlap_loader = MagicMock()
        scheduler.tp_worker = MagicMock()
        scheduler.disaggregation_mode = DisaggregationMode.NONE
        scheduler.req_to_token_pool = MagicMock()
        scheduler.enable_hicache_storage = False
        scheduler.truncation_align_size = None
        scheduler._add_request_to_queue = MagicMock()
        scheduler.predict_next_chunk_size = MagicMock()
        scheduler.get_num_allocatable_reqs = types.MethodType(
            lambda self, running_bs: 1, scheduler
        )
        return scheduler

    def test_prefill_budget_subtracts_decode_count(self):
        scheduler = self._make_scheduler(
            running_bs=3,
            max_prefill_tokens=100,
            max_tokens_per_iteration=10,
        )
        with patch.object(scheduler_mod, "PrefillAdder", _FakePrefillAdder):
            scheduler._get_new_batch_prefill_raw(prefill_delayer_single_pass=None)
        self.assertEqual(_FakePrefillAdder.last_rem_input_tokens, 7)

    def test_prefill_budget_clamped_by_max_prefill_tokens(self):
        scheduler = self._make_scheduler(
            running_bs=1,
            max_prefill_tokens=4,
            max_tokens_per_iteration=10,
        )
        with patch.object(scheduler_mod, "PrefillAdder", _FakePrefillAdder):
            scheduler._get_new_batch_prefill_raw(prefill_delayer_single_pass=None)
        self.assertEqual(_FakePrefillAdder.last_rem_input_tokens, 4)


if __name__ == "__main__":
    unittest.main(verbosity=3)
