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

    def test_cache_read_billed_at_nominal_rate(self):
        # NOMINAL published read multiplier 0.1x (R2) -- reproduces the CLI's
        # per-model costUSD exactly; see _usage._CACHE_READ_MULT.
        cost = cost_for_usage("claude-opus-4-8", {"cache_read_input_tokens": 1_000_000})
        assert cost == 5.0 * 0.1

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
        # NOMINAL write multipliers: 5m=1.25x, 1h=2.0x (R2).
        assert cost == 5.0 * 1.25 + 5.0 * 2.0

    def test_flat_cache_creation_billed_at_5m_write_rate(self):
        # A flat cache_creation_input_tokens (no 5m/1h split) is billed at the
        # nominal 5-minute write rate.
        cost = cost_for_usage(
            "claude-opus-4-8", {"cache_creation_input_tokens": 1_000_000}
        )
        assert cost == 5.0 * 1.25

    def test_reproduces_live_ground_truth(self):
        """Nominal rates reproduce the CLI's per-model costUSD EXACTLY (R2).

        Ground truth: the live old-SDK ResultMessage ``model_usage`` per-model
        ``costUSD`` (the opus component, which is what cost_for_usage models).
        Captured live (claude-opus-4-8, PONG, acceptEdits):
        usage input=2, output=5, cache_read=16122, cache_creation 5m=1891 ->
        costUSD=0.02001475. Nominal rates reproduce it to 0.00%. (The previous
        calibrated multipliers over-counted this opus component by ~5.5% because
        they were fit to the result-level total, which bundles the unobservable
        haiku title-gen line -- see R2/R3.)
        """
        usage = {
            "input_tokens": 2,
            "output_tokens": 5,
            "cache_read_input_tokens": 16122,
            "cache_creation_input_tokens": 1891,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 1891,
                "ephemeral_1h_input_tokens": 0,
            },
        }
        computed = cost_for_usage("claude-opus-4-8", usage)
        assert computed is not None
        real = 0.02001475
        # Exact to floating-point tolerance (<0.5% per R2; in practice ~0%).
        assert abs(computed - real) / real < 0.005, f"computed {computed} vs {real}"

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
        # camelCase sub-keys matching the real CLI wire format / baseline (R1).
        assert mu["claude-opus-4-8"]["inputTokens"] == 1_000_000
        assert mu["claude-opus-4-8"]["costUSD"] == 5.0
        assert mu["claude-sonnet-4-6"]["costUSD"] == 3.0
        # contextWindow / maxOutputTokens surfaced like the baseline.
        assert mu["claude-opus-4-8"]["contextWindow"] == 1_000_000
        assert mu["claude-opus-4-8"]["maxOutputTokens"] == 64_000
        # The full baseline camelCase key set is present.
        assert set(mu["claude-opus-4-8"]) == {
            "inputTokens",
            "outputTokens",
            "cacheReadInputTokens",
            "cacheCreationInputTokens",
            "webSearchRequests",
            "costUSD",
            "contextWindow",
            "maxOutputTokens",
        }

    def test_empty_accumulator(self):
        acc = TurnUsageAccumulator()
        assert acc.has_data() is False
        assert acc.aggregate_usage() is None
        assert acc.total_cost() is None
        assert acc.model_usage() is None
