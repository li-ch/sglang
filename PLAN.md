# Throughput-Maximized LLM Serving - Development Plan

## Goal
Modify SGLang to maximize throughput (tokens/sec/GPU) for non-streaming,
batch-oriented workloads (AI agents). We do NOT care about latency (TTFT or TPOT).

## Key Architectural Changes

### 1. Scheduler Modifications (Step 2 - HIGHEST PRIORITY)
- Remove all latency-oriented scheduling logic
- Remove TTFT-driven prefill prioritization
- Implement greedy token packing: each iteration fills to a compute-saturation
  threshold τ with as many tokens as possible
  - Implemented: prefill budget is greedy within τ after subtracting decode tokens
- New requests don't interrupt decode batches; prefill tokens fill slack capacity
- Batch size limited only by GPU memory, not latency targets
  - Implemented: per-iteration request caps driven by latency removed
- Adaptive τ: if last iteration was fast, increase τ; if slow, decrease τ
  - Implemented: τ budget subtracts active decode count each iteration, fills
    remaining budget with prefill tokens, preferring longer cached prefixes
  - Implemented: τ feedback loop adjusts by ±5% vs target iteration time
    estimated from the first 10 iterations (with min/max clamps)

### 2. Memory Management (Step 3)
- Increase memory utilization target to 95%
- Simple preemption: pause lowest-priority requests, offload KV to CPU
- Output length prediction for admission control

### 3. Prefix Cache Tuning (Step 3)
- Optimize eviction for agent patterns (shared system prompts)
- Prefer keeping long prefixes over short ones

### 4. API Layer (Step 4)
- Non-streaming batch API
- Priority levels per request

## Key Files to Modify
- python/sglang/srt/managers/scheduler.py (main scheduling loop)
- python/sglang/srt/managers/schedule_batch.py (batch structures)
- python/sglang/srt/server_args.py (τ config flags and bounds)
- python/sglang/srt/mem_cache/memory_pool.py (memory management)
- python/sglang/srt/mem_cache/radix_cache.py (prefix cache)
- test/registered/scheduler/test_tau_prefill_budget.py (τ budget unit test)
- test/registered/scheduler/test_tau_feedback_loop.py (τ feedback loop unit test)

## Scheduler Loop Notes (Current Behavior)
- Main loop (`event_loop_normal` / `event_loop_overlap`) does: recv requests → `get_next_batch_to_run()` → `run_batch()` → process result. Prefill and decode are chosen inside `get_next_batch_to_run`.
- Prefill vs decode decision: `get_next_batch_to_run` constructs a prefill batch via `get_new_batch_prefill()`. If a new batch exists, it runs prefill; otherwise it updates/uses the running decode batch (`update_running_batch`) and returns that for decode.
- Prefill sizing: `_get_new_batch_prefill_raw` sets `chunked_prefill_size` (static or dynamic) and builds a `PrefillAdder` with a τ-based per-iteration budget, `chunked_prefill_size`, `new_token_ratio`, `max_prefill_bs`, and `prefill_max_requests`, which governs how many prefill tokens/requests can be added in that iteration.
- Latency-related constraints: waiting/running timeouts (`SGLANG_REQ_WAITING_TIMEOUT`, `SGLANG_REQ_RUNNING_TIMEOUT`) abort requests; `PrefillDelayer` can delay prefill based on token usage; overlap is disabled for back-to-back prefills to improve TTFT; watchdog/soft watchdog timeouts exist.

## Current SGLang Behavior
- Max batch size is bounded by request-pool availability: `get_num_allocatable_reqs` now returns `req_to_token_pool.available_size()` (ignoring `pp_max_micro_batch_size`) so additions are limited by free request slots rather than per-iteration caps.
- `ScheduleBatch.batch_size()` is simply `len(reqs)`; scheduling limits come from the scheduler’s allocatable-reqs logic, not from `ScheduleBatch` itself.
- Prefix cache eviction is policy-driven (`lru`, `lfu`, `fifo`, `mru`, `filo`, `priority`). `RadixCache.evict` builds a heap of evictable leaves ordered by the strategy’s priority and evicts leaf nodes first; parent nodes become candidates if they become leaf and are unlocked.
- Memory pools track free blocks via `free_slots`: `ReqToTokenPool` uses a Python list of free request slots; `MambaPool` uses a tensor of free indices. `alloc(...)` returns `None` when `need_size > len(free_slots)`, which is how memory pressure manifests (and upstream scheduling logic uses available size/failed alloc to stop admitting new requests).

## Latency-Oriented Limits Removed (Throughput Defaults)
- **Chunked prefill size**: default is now `-1` (disabled). Chunking only happens if explicitly enabled, so prefill tokens aren’t capped per iteration by default.
- **Prefill delayer**: env-compat auto-enabling is disabled unless `enable_prefill_delayer` is already true; default remains off so prefill is not delayed for TTFT.
- **Prefill request cap**: `prefill_max_requests` remains `None` by default, so no hard per-iteration request cap is applied.
- **Per-iteration request cap (PP)**: `get_num_allocatable_reqs()` no longer uses `pp_max_micro_batch_size` and relies on request-pool availability instead.
- **TTFT overlap guard**: consecutive prefill overlap is no longer disabled; overlap is only turned off for grammar sync constraints.

## Constraints
- Stay in Python for prototype speed
- Don't modify CUDA kernels
- Don't modify model loading or forward pass logic
- All changes should be in the scheduling and memory management layers