"""Unit tests for PTY transport usage aggregation and cost computation.

These are dependency-free (no live model / no subprocess): they exercise the
``_usage`` module that reconstructs the stream-json ``ResultMessage`` cost /
usage / model_usage fields from per-assistant-message transcript usage dicts.
"""

from claude_agent_sdk._internal.transport._usage import (
    TurnUsageAccumulator,
    cost_for_usage,
)


class TestCostForUsage:
    def test_unknown_model_returns_none(self):
        assert cost_for_usage("mystery-model", {"input_tokens": 100}) is None
        assert cost_for_usage(None, {"input_tokens": 100}) is None

    def test_opus_pricing(self):
        # opus: input $5/Mtok, output $25/Mtok
        cost = cost_for_usage(
            "claude-opus-4-8", {"input_tokens": 1_000_000, "output_tokens": 0}
        )
        assert cost == 5.0
        cost = cost_for_usage(
            "claude-opus-4-8", {"input_tokens": 0, "output_tokens": 1_000_000}
        )
        assert cost == 25.0

    def test_sonnet_and_haiku_pricing(self):
        assert cost_for_usage("claude-sonnet-4-6", {"input_tokens": 1_000_000}) == 3.0
        assert cost_for_usage("claude-haiku-4-5", {"input_tokens": 1_000_000}) == 1.0

    def test_dated_and_aliased_model_ids_resolve(self):
        assert (
            cost_for_usage("claude-opus-4-5-20251101", {"input_tokens": 1_000_000})
            == 5.0
        )

    def test_cache_read_billed_at_calibrated_rate(self):
        # Calibrated read multiplier (~0.1054) fit to live ground truth, not the
        # nominal 0.1 -- see _usage._CACHE_READ_MULT.
        cost = cost_for_usage("claude-opus-4-8", {"cache_read_input_tokens": 1_000_000})
        assert cost == 5.0 * 0.10543

    def test_cache_creation_split_5m_1h(self):
        cost = cost_for_usage(
            "claude-opus-4-8",
            {
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 1_000_000,
                    "ephemeral_1h_input_tokens": 1_000_000,
                }
            },
        )
        # Both buckets use the calibrated write multiplier (~1.3215).
        assert cost == 5.0 * 1.32145 + 5.0 * 1.32145

    def test_flat_cache_creation_billed_at_write_rate(self):
        cost = cost_for_usage(
            "claude-opus-4-8", {"cache_creation_input_tokens": 1_000_000}
        )
        assert cost == 5.0 * 1.32145

    def test_reproduces_live_ground_truth(self):
        """The calibrated multipliers reproduce real stream-json totals.

        Four single-API-call ResultMessage points captured live from the old
        stream-json transport (claude-opus-4-8, acceptEdits). cost_for_usage on
        each message's usage must match total_cost_usd within a tight tolerance.
        """
        points = [
            # input, output, cache_read, cache_creation_1h, real_total_cost_usd
            (1893, 4, 15914, 207, 0.01932175),
            (1893, 10, 15914, 224, 0.019577),
            (1893, 5, 15914, 208, 0.019364),
            (1893, 4, 15914, 11423, 0.09342875),
        ]
        for inp, out, cr, cc, real in points:
            usage = {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": cr,
                "cache_creation_input_tokens": cc,
                "cache_creation": {
                    "ephemeral_1h_input_tokens": cc,
                    "ephemeral_5m_input_tokens": 0,
                },
            }
            computed = cost_for_usage("claude-opus-4-8", usage)
            assert computed is not None
            # Within 0.1% of the live total.
            assert abs(computed - real) / real < 0.001, (
                f"cc={cc}: computed {computed} vs real {real}"
            )

    def test_bool_tokens_ignored(self):
        # Guards against True being treated as 1.
        assert cost_for_usage("claude-opus-4-8", {"input_tokens": True}) == 0.0


class TestTurnUsageAccumulator:
    def test_dedup_by_message_id_does_not_double_count(self):
        acc = TurnUsageAccumulator()
        usage = {"input_tokens": 100, "output_tokens": 10}
        # Same message id streamed three times -> counted once.
        acc.add("msg_1", "claude-opus-4-8", usage)
        acc.add("msg_1", "claude-opus-4-8", usage)
        acc.add("msg_1", "claude-opus-4-8", usage)
        agg = acc.aggregate_usage()
        assert agg["input_tokens"] == 100
        assert agg["output_tokens"] == 10

    def test_distinct_message_ids_sum(self):
        acc = TurnUsageAccumulator()
        acc.add("msg_1", "claude-opus-4-8", {"input_tokens": 100})
        acc.add("msg_2", "claude-opus-4-8", {"input_tokens": 50})
        assert acc.aggregate_usage()["input_tokens"] == 150

    def test_latest_snapshot_per_id_wins(self):
        acc = TurnUsageAccumulator()
        acc.add("msg_1", "claude-opus-4-8", {"input_tokens": 1})
        acc.add("msg_1", "claude-opus-4-8", {"input_tokens": 100})
        assert acc.aggregate_usage()["input_tokens"] == 100

    def test_anonymous_usage_dicts_sum(self):
        acc = TurnUsageAccumulator()
        acc.add(None, "m", {"input_tokens": 10})
        acc.add(None, "m", {"input_tokens": 5})
        assert acc.aggregate_usage()["input_tokens"] == 15

    def test_nested_cache_creation_merged(self):
        acc = TurnUsageAccumulator()
        acc.add(
            "msg_1",
            "claude-opus-4-8",
            {"cache_creation": {"ephemeral_5m_input_tokens": 10}},
        )
        acc.add(
            "msg_2",
            "claude-opus-4-8",
            {"cache_creation": {"ephemeral_5m_input_tokens": 20}},
        )
        agg = acc.aggregate_usage()
        assert agg["cache_creation"]["ephemeral_5m_input_tokens"] == 30

    def test_total_cost_sums_distinct_messages(self):
        acc = TurnUsageAccumulator()
        acc.add("msg_1", "claude-opus-4-8", {"input_tokens": 1_000_000})
        acc.add("msg_1", "claude-opus-4-8", {"input_tokens": 1_000_000})  # dup
        acc.add("msg_2", "claude-opus-4-8", {"output_tokens": 1_000_000})
        # 5.0 (msg_1 once) + 25.0 (msg_2) -- dup not counted.
        assert acc.total_cost() == 30.0

    def test_total_cost_none_when_all_models_unknown(self):
        acc = TurnUsageAccumulator()
        acc.add("msg_1", "mystery", {"input_tokens": 100})
        assert acc.total_cost() is None

    def test_total_cost_sums_known_when_mixed(self):
        acc = TurnUsageAccumulator()
        acc.add("msg_1", "mystery", {"input_tokens": 1_000_000})
        acc.add("msg_2", "claude-opus-4-8", {"input_tokens": 1_000_000})
        assert acc.total_cost() == 5.0

    def test_model_usage_breakdown(self):
        acc = TurnUsageAccumulator()
        acc.add("msg_1", "claude-opus-4-8", {"input_tokens": 1_000_000})
        acc.add("msg_2", "claude-sonnet-4-6", {"input_tokens": 1_000_000})
        mu = acc.model_usage()
        assert set(mu.keys()) == {"claude-opus-4-8", "claude-sonnet-4-6"}
        assert mu["claude-opus-4-8"]["input_tokens"] == 1_000_000
        assert mu["claude-opus-4-8"]["cost_usd"] == 5.0
        assert mu["claude-sonnet-4-6"]["cost_usd"] == 3.0

    def test_empty_accumulator(self):
        acc = TurnUsageAccumulator()
        assert acc.has_data() is False
        assert acc.aggregate_usage() is None
        assert acc.total_cost() is None
        assert acc.model_usage() is None
