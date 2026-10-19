"""Tests for analytic roofline math, config-space filtering, and persistence."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from flashattn_tile import autotune as at


def test_flops_causal_and_noncausal():
    # 4 * H * D * pairs
    assert at.flops_attention(1, 4, 2, 3, causal=False) == 4 * 1 * 4 * 6
    # pairs = 2 * (3 - 2) + 2 * 3 // 2 = 5
    assert at.flops_attention(1, 4, 2, 3, causal=True) == 4 * 1 * 4 * 5
    assert at.flops_attention(8, 128, 1, 4096, causal=True, batch=4) == (
        4 * 4 * 8 * 128 * 4096
    )


def test_flops_varlen_sums_sequences():
    total = at.flops_attention_varlen(2, 8, [1, 4], [10, 4], causal=True)
    expected = at.flops_attention(2, 8, 1, 10, causal=True) + at.flops_attention(
        2, 8, 4, 4, causal=True
    )
    assert total == expected
    with pytest.raises(ValueError):
        at.flops_attention_varlen(2, 8, [1], [1, 2])


def test_bytes_decode_dominated_by_kv():
    nbytes = at.bytes_attention("decode", 8, 1, 4096, 32, 8, 128, elem_bytes=2)
    kv = 8 * 4096 * 8 * 128 * 2 * 2
    q = 8 * 32 * 128 * 2
    assert nbytes == kv + 2 * q
    ai = at.arithmetic_intensity(
        at.flops_attention(32, 128, 1, 4096, causal=True, batch=8), nbytes
    )
    # 4 FLOPs per KV pair, 4 bytes per KV pair, times GQA group size 4 -> ~4.
    assert 3.0 < ai < 5.0


def test_bytes_prefill_and_invalid_mode():
    nbytes = at.bytes_attention("prefill", 1, 1024, 1024, 32, 8, 128, 2)
    expected = 2 * (1024 * 32 * 128 * 2) + 1024 * 8 * 128 * 2 * 2
    assert nbytes == expected
    with pytest.raises(ValueError):
        at.bytes_attention("backward", 1, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        at.bytes_attention("decode", 1, 2, 4, 1, 1, 1)


def test_config_space_counts_and_smem_filtering():
    prefill = at.build_config_space("prefill", 128, 2, smem_limit=10**9)
    decode = at.build_config_space("decode", 128, 2, page_size=16, smem_limit=10**9)
    assert len(prefill) == 3 * 3 * 2 * 2
    assert len(decode) == 2 * 2 * 4
    assert all(c.smem_bytes > 0 for c in prefill.candidates)

    tight = at.build_config_space("decode", 128, 2, page_size=16, smem_limit=20_000)
    feasible = tight.feasible()
    assert 0 < len(feasible) < len(tight)
    assert all(c.smem_bytes <= 20_000 for c in feasible)
    assert tight.to_dict()["feasible"] == len(feasible)

    none = at.build_config_space("decode", 128, 2, page_size=16, smem_limit=1)
    assert none.feasible() == []


def test_smem_grows_with_stages():
    two = at.prefill_smem_bytes(64, 64, 128, 2, 2)
    three = at.prefill_smem_bytes(64, 64, 128, 2, 3)
    assert three > two
    assert at.decode_smem_bytes(16, 128, 2, 2) < at.decode_smem_bytes(32, 128, 2, 2)


def test_autotune_ranks_and_reports_metrics():
    calls = {"n": 0}

    def runner(cfg):
        calls["n"] += 1
        # Cheap deterministic work; only relative ordering matters.
        for _ in range(cfg["split_k"] * 100):
            pass

    candidates = [
        at.TuneConfig(1, 16, 4, 2, 1, 16_448),
        at.TuneConfig(1, 16, 4, 3, 8, 24_640),
    ]
    problem = {
        "kind": "decode",
        "signature": "unit-test",
        "runner": runner,
        "flops": 1_000_000,
        "bytes": 2_000_000,
    }
    df = at.autotune(problem, candidates, warmup=1, iters=3, hbm_gbs=50.0, save=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]["latency_ms"] <= df.iloc[1]["latency_ms"]
    assert df.iloc[0]["rank"] == 1
    for column in ("tflops", "gbytes_per_s", "pct_measured_hbm", "arithmetic_intensity"):
        assert column in df
    assert df.iloc[0]["arithmetic_intensity"] == pytest.approx(0.5)
    assert calls["n"] == 2 * (1 + 3)


def test_save_and_load_best_config(tmp_path, monkeypatch):
    path = tmp_path / "tuned_configs.json"
    monkeypatch.setattr(at, "TUNED_CONFIGS_PATH", path)
    assert at.best_config("decode", "sig") is None
    cfg = {"BLOCK_M": 1, "BLOCK_N": 16, "num_warps": 4, "num_stages": 2, "split_k": 4}
    at.save_best("decode", "sig", cfg, {"latency_ms": 0.5})
    assert at.best_config("decode", "sig") == cfg
    stored = json.loads(path.read_text())
    assert stored["configs"]["decode:sig"]["metrics"]["latency_ms"] == 0.5


def test_time_callable_rejects_bad_iters():
    with pytest.raises(ValueError):
        at.time_callable(lambda: None, warmup=0, iters=1)
