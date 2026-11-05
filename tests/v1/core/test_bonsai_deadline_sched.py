# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the opt-in decode-deadline prefill token budget."""

import pytest

from vllm.config.scheduler import SchedulerConfig, parse_cost_model
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler

PROMPT = 200


def _make(deadline_ms: float | None, min_chunk: int = 16) -> Scheduler:
    """Scheduler with a=30,b=1,c=1,d=0 and the given deadline."""
    scheduler = create_scheduler()
    scheduler.decode_deadline_ms = deadline_ms
    scheduler.cost_model = (30.0, 1.0, 1.0, 0.0)
    object.__setattr__(scheduler.scheduler_config, "min_prefill_chunk", min_chunk)
    return scheduler


def _start_decoders(scheduler: Scheduler, n: int) -> None:
    """Prefill n short requests and move them into the decode state."""
    ids = [f"run{i}" for i in range(n)]
    for req in create_requests(
        num_requests=n, num_tokens=8, max_tokens=100, req_ids=ids
    ):
        scheduler.add_request(req)
    output = scheduler.schedule()
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[0] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _prefill_tokens(scheduler: Scheduler) -> int:
    """Add one long prompt and return the tokens scheduled for it."""
    (req,) = create_requests(num_requests=1, num_tokens=PROMPT, req_ids=["new"])
    scheduler.add_request(req)
    return scheduler.schedule().num_scheduled_tokens["new"]


def test_no_decode_full_budget():
    assert _prefill_tokens(_make(60.0)) == PROMPT


def test_three_decoders_capped_chunk():
    scheduler = _make(60.0)
    _start_decoders(scheduler, 3)
    # cap = (60 - 30 - 1*3) / 1 = 27
    assert _prefill_tokens(scheduler) == 27


def test_min_chunk_respected():
    scheduler = _make(20.0, min_chunk=16)
    _start_decoders(scheduler, 3)
    assert _prefill_tokens(scheduler) == 16


def test_unset_identical_to_default():
    outs = []
    for deadline in (None, None):
        scheduler = _make(deadline)
        _start_decoders(scheduler, 3)
        outs.append(_prefill_tokens(scheduler))
    assert outs == [PROMPT, PROMPT]


def test_cost_model_validation():
    assert parse_cost_model("1,2,3,4") == (1.0, 2.0, 3.0, 4.0)
    for bad in ("1,2,3", "a,b,c,d", "1,2,3,nan"):
        with pytest.raises(ValueError):
            parse_cost_model(bad)
    with pytest.raises(ValueError):
        SchedulerConfig(decode_deadline_ms=0)


class _Clock:
    """Fake perf_counter advanced manually."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _auto(deadline_ms: float) -> tuple[Scheduler, _Clock]:
    """Scheduler in feedback mode with a fake clock."""
    scheduler = _make(deadline_ms)
    scheduler.cost_model = None
    scheduler._clock = clock = _Clock()
    return scheduler, clock


def test_auto_cap_shrinks_when_slow_and_grows_when_fast():
    scheduler, clock = _auto(50.0)
    _start_decoders(scheduler, 3)
    first = scheduler._deadline_prefill_cap()
    clock.t += 0.100  # previous step took 100 ms -> ratio 0.5
    assert scheduler._deadline_prefill_cap() == max(16, int(first * 0.5))
    cap = scheduler._auto_cap
    clock.t += 0.010  # 10 ms -> ratio clamped to 1.25
    assert scheduler._deadline_prefill_cap() == int(cap * 1.25)


def test_auto_cap_bounds_and_reset():
    scheduler, clock = _auto(50.0)
    _start_decoders(scheduler, 3)
    scheduler._deadline_prefill_cap()
    for _ in range(20):
        clock.t += 1.0
        assert scheduler._deadline_prefill_cap() == 16
    scheduler.running = []
    assert scheduler._deadline_prefill_cap() is None
    assert scheduler._auto_cap is None and scheduler._last_sched_time is None
