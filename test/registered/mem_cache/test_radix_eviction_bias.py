import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestRadixEvictionBias(CustomTestCase):
    def test_evict_prefers_shorter_prefix_on_tie(self):
        allocator = MagicMock()
        cache = RadixCache.create_simulated(mock_allocator=allocator, page_size=1)

        short_key = RadixKey([1, 2])
        long_key = RadixKey([3, 4, 5, 6])

        cache.insert(
            InsertParams(
                key=short_key,
                value=torch.tensor(short_key.token_ids, dtype=torch.int64),
                priority=0,
            )
        )
        cache.insert(
            InsertParams(
                key=long_key,
                value=torch.tensor(long_key.token_ids, dtype=torch.int64),
                priority=0,
            )
        )

        for node in cache.evictable_leaves:
            node.last_access_time = 1.0

        cache.evict(EvictParams(num_tokens=2))

        remaining_lengths = sorted(len(node.key) for node in cache.evictable_leaves)
        self.assertEqual(remaining_lengths, [4])


if __name__ == "__main__":
    unittest.main(verbosity=3)
