"""Usage aggregation and client-side cost computation for the PTY transport.

The interactive transcript does **not** record cost, ``modelUsage``, or a single
result-level ``usage`` block the way the deleted stream-json transport's
``ResultMessage`` did. It only records per-assistant-message ``usage`` dicts. To
be a faithful drop-in we must reconstruct those result fields from the
transcript:

* **usage** -- the per-turn token totals. The transcript writes *multiple
  snapshots of the same assistant message* (same ``message.id``) as it streams,
  so naively summing every assistant record over-counts cache tokens massively.
  We dedup by ``message.id`` (keeping the final snapshot per id) and sum the
  distinct messages of the turn. Verified against captured transcripts where the
  three snapshots of one id carry identical usage.
* **total_cost_usd** -- not in the transcript at all. We compute it from the
  per-message usage and a small model-pricing table. The CLI derives cost
  per-API-iteration; for the common single-iteration assistant messages this is
  exactly ``sum(per_message_cost)``.
* **model_usage** -- a per-model breakdown keyed by model id (mirrors the
  stream-json ``modelUsage`` field), each value carrying that model's token
  totals and computed cost.

Pricing table (USD per million tokens) is small and clearly marked for
maintenance. Cache multipliers are calibrated against live stream-json
``total_cost_usd`` ground truth (see ``_CACHE_*_MULT`` below), not just the
nominal published rates -- the published 0.1x read / 1.25x 5m-write / 2x
1h-write rates over-estimate the CLI's billed cost by ~1.4%, so the multipliers
here are fit to reproduce the real result to within ~0.06%.
"""

from __future__ import annotations

from typing import Any

# USD per 1,000,000 tokens. Keyed by a coarse model family matched as a
# substring of the model id (so "claude-opus-4-8", "claude-opus-4-5-20251101",
# etc. all resolve to the opus row). MAINTENANCE: update when pricing changes or
# a new family ships. Sources: Anthropic model pricing.
_PRICING_PER_MTOK: dict[str, dict[str, float]] = {
    "opus": {"input": 5.0, "output": 25.0},
    "sonnet": {"input": 3.0, "output": 15.0},
    "haiku": {"input": 1.0, "output": 5.0},
}

# Cache multipliers relative to the model's base input price.
#
# These are CALIBRATED to live stream-json ``total_cost_usd`` ground truth, not
# the nominal published rates. Solving four live single-API-call result points
# (same model, varying cache_creation so the read term cancels) gives an
# effective read multiplier of ~0.1054 and an effective 1h-write multiplier of
# ~1.3215 -- each ~5.5% above the nominal 0.1 / 1.25 ("ephemeral_1h" tokens are
# billed close to the 5-minute write rate here, not the nominal 2x). The
# nominal rates over-estimate by ~1.4%; these reproduce the four points to
# within ~0.06%. The 5m-write multiplier tracks the 1h one at the same ~5.5%
# offset over its 1.25 nominal (no live 5m-only data point is available to
# separate them, and in practice the CLI writes all cache_creation to one
# bucket per message). MAINTENANCE: re-fit if billed pricing changes.
_CACHE_READ_MULT = 0.10543
_CACHE_WRITE_5M_MULT = 1.32145
_CACHE_WRITE_1H_MULT = 1.32145

# Token-count usage keys we sum when aggregating a turn's usage. Nested dicts
# (cache_creation, server_tool_use) and non-numeric metadata are merged
# separately so we never coerce a dict into an int.
_NUMERIC_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


def _pricing_for(model: str | None) -> dict[str, float] | None:
    if not model:
        return None
    low = model.lower()
    for family, prices in _PRICING_PER_MTOK.items():
        if family in low:
            return prices
    return None


def cost_for_usage(model: str | None, usage: dict[str, Any]) -> float | None:
    """Compute the USD cost of a single usage dict for ``model``.

    Returns ``None`` when the model is unknown (so callers can decide whether to
    fall back) and ``0.0`` for a known model with no billable tokens. Cache
    creation is split into 5-minute and 1-hour buckets when the transcript
    provides ``cache_creation``; otherwise the flat
    ``cache_creation_input_tokens`` is billed at the 5-minute rate.
    """
    prices = _pricing_for(model)
    if prices is None:
        return None
    in_rate = prices["input"] / 1_000_000
    out_rate = prices["output"] / 1_000_000

    input_tokens = _as_int(usage.get("input_tokens"))
    output_tokens = _as_int(usage.get("output_tokens"))
    cache_read = _as_int(usage.get("cache_read_input_tokens"))

    cache_creation = usage.get("cache_creation")
    if isinstance(cache_creation, dict):
        write_5m = _as_int(cache_creation.get("ephemeral_5m_input_tokens"))
        write_1h = _as_int(cache_creation.get("ephemeral_1h_input_tokens"))
    else:
        write_5m = _as_int(usage.get("cache_creation_input_tokens"))
        write_1h = 0

    cost = (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_read * in_rate * _CACHE_READ_MULT
        + write_5m * in_rate * _CACHE_WRITE_5M_MULT
        + write_1h * in_rate * _CACHE_WRITE_1H_MULT
    )
    return cost


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    return 0


class TurnUsageAccumulator:
    """Aggregates per-assistant-message usage for one turn.

    Dedups by ``message.id``: the transcript rewrites the same assistant message
    several times while it streams, and every snapshot carries the same usage,
    so we keep only the latest snapshot per id and sum distinct ids. This avoids
    the over-counting a naive sum across raw assistant records produces.
    """

    def __init__(self) -> None:
        # message_id -> (model, usage dict from the latest snapshot)
        self._by_message: dict[str, tuple[str | None, dict[str, Any]]] = {}
        # Counter for usage dicts that have no message id (rare); keyed
        # positionally so distinct anonymous messages are not collapsed.
        self._anon: list[tuple[str | None, dict[str, Any]]] = []

    def add(self, message_id: str | None, model: str | None, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        if message_id:
            # Latest snapshot wins (final usage for the message).
            self._by_message[message_id] = (model, usage)
        else:
            self._anon.append((model, usage))

    def _entries(self) -> list[tuple[str | None, dict[str, Any]]]:
        return list(self._by_message.values()) + self._anon

    def has_data(self) -> bool:
        return bool(self._by_message) or bool(self._anon)

    def aggregate_usage(self) -> dict[str, Any] | None:
        """Return the summed usage block across the turn's distinct messages."""
        entries = self._entries()
        if not entries:
            return None
        total: dict[str, Any] = {}
        nested_cache: dict[str, int] = {}
        server_tool: dict[str, int] = {}
        for _model, usage in entries:
            for key in _NUMERIC_USAGE_KEYS:
                if key in usage:
                    total[key] = total.get(key, 0) + _as_int(usage.get(key))
            cc = usage.get("cache_creation")
            if isinstance(cc, dict):
                for k, v in cc.items():
                    nested_cache[k] = nested_cache.get(k, 0) + _as_int(v)
            stu = usage.get("server_tool_use")
            if isinstance(stu, dict):
                for k, v in stu.items():
                    server_tool[k] = server_tool.get(k, 0) + _as_int(v)
            # Carry through stable metadata (service_tier, etc.) from the first
            # message that has it.
            for k, v in usage.items():
                if (
                    k not in _NUMERIC_USAGE_KEYS
                    and k not in ("cache_creation", "server_tool_use", "iterations")
                    and k not in total
                ):
                    total[k] = v
        if nested_cache:
            total["cache_creation"] = nested_cache
        if server_tool:
            total["server_tool_use"] = server_tool
        return total

    def total_cost(self) -> float | None:
        """Sum the computed cost across distinct messages, or ``None``.

        ``None`` only when *no* message had a known model (so we have no basis
        to compute any cost); a mix of known/unknown sums the known ones.
        """
        entries = self._entries()
        if not entries:
            return None
        total = 0.0
        any_known = False
        for model, usage in entries:
            c = cost_for_usage(model, usage)
            if c is not None:
                any_known = True
                total += c
        return total if any_known else None

    def model_usage(self) -> dict[str, Any] | None:
        """Per-model token + cost breakdown, mirroring stream-json ``modelUsage``."""
        entries = self._entries()
        if not entries:
            return None
        out: dict[str, Any] = {}
        for model, usage in entries:
            key = model or "unknown"
            bucket = out.setdefault(
                key,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            for k in _NUMERIC_USAGE_KEYS:
                bucket[k] += _as_int(usage.get(k))
            c = cost_for_usage(model, usage)
            if c is not None:
                bucket["cost_usd"] += c
        return out
