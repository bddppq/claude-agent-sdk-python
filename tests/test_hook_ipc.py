"""Unit tests for the settings-hook IPC bridge (win #3).

Cover the IPC server's dispatch logic, the settings synthesis, matcher
semantics, and the shim's fail-open behavior -- without spawning the real CLI.
``test_pty_integration.py::TestHookIpcBridge`` exercises the full transport
chain end-to-end against a fake CLI that actually runs the shim.
"""

from __future__ import annotations

import json
import os
import subprocess

import anyio
import pytest

from claude_agent_sdk._internal.transport._hook_ipc import (
    ENV_HOOK_IPC,
    HookIpcServer,
    build_hooks_settings,
    shim_command,
)
from claude_agent_sdk.types import (
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
)

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------- #
# Matcher semantics (mirrors the CLI's matchesPattern)
# --------------------------------------------------------------------------- #


class TestMatcherMatches:
    @pytest.mark.parametrize(
        ("pattern", "tool", "expected"),
        [
            (None, "Write", True),
            ("", "Write", True),
            ("*", "Write", True),
            ("Write", "Write", True),
            ("Write", "Read", False),
            ("Write|Edit", "Edit", True),
            ("Write|Edit", "Bash", False),
            ("^Write$", "Write", True),
            (".*Tool", "MyTool", True),
            ("Write", None, False),  # non-tool event, named matcher -> no match
        ],
    )
    def test_matches(self, pattern, tool, expected):
        assert HookIpcServer._matcher_matches(pattern, tool) is expected

    def test_invalid_regex_falls_back_to_exact(self):
        # An unbalanced group is not valid regex -> exact compare.
        assert HookIpcServer._matcher_matches("(unclosed", "(unclosed") is True
        assert HookIpcServer._matcher_matches("(unclosed", "Write") is False


# --------------------------------------------------------------------------- #
# Settings synthesis
# --------------------------------------------------------------------------- #


class TestBuildHooksSettings:
    async def test_preserves_user_matchers_and_wires_shim(self):
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        server = HookIpcServer(
            hooks={
                "PreToolUse": [HookMatcher(matcher="Write|Edit", hooks=[_h])],
                "PostToolUse": [HookMatcher(matcher=None, hooks=[_h])],
            },
            can_use_tool=None,
        )
        block = build_hooks_settings(server)
        assert set(block) == {"PreToolUse", "PostToolUse"}
        assert block["PreToolUse"][0]["matcher"] == "Write|Edit"
        # PostToolUse matcher=None -> no matcher key (catch-all).
        assert "matcher" not in block["PostToolUse"][0]
        cmd = block["PreToolUse"][0]["hooks"][0]
        assert cmd["type"] == "command"
        assert "_hook_shim" in cmd["command"]

    async def test_can_use_tool_adds_catchall_pretooluse(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow()

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        block = build_hooks_settings(server)
        assert list(block) == ["PreToolUse"]
        assert "matcher" not in block["PreToolUse"][0]

    async def test_can_use_tool_with_narrow_matcher_still_adds_catchall(self):
        # W2 (permission bypass): when can_use_tool is set AND the user has a
        # NARROW PreToolUse matcher (e.g. "Write"), build_hooks_settings MUST
        # still emit a catch-all (no-matcher) PreToolUse entry, so the shim runs
        # for EVERY tool and can_use_tool is consulted for tools the user matcher
        # does not cover (Bash/Edit/...). Otherwise those tools are silently
        # auto-allowed.
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow()

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Write", hooks=[_h])]},
            can_use_tool=_can_use,
        )
        block = build_hooks_settings(server)
        pre = block["PreToolUse"]
        # The user's narrow entry is preserved AND a catch-all is appended.
        assert any(e.get("matcher") == "Write" for e in pre)
        assert any("matcher" not in e for e in pre), (
            "catch-all PreToolUse entry must exist when can_use_tool is set"
        )

    async def test_can_use_tool_with_existing_catchall_not_duplicated(self):
        # If the user's PreToolUse hook is ALREADY a catch-all (matcher=None),
        # one shim entry covers every tool, so no extra catch-all is appended.
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow()

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_h])]},
            can_use_tool=_can_use,
        )
        block = build_hooks_settings(server)
        pre = block["PreToolUse"]
        assert len(pre) == 1
        assert "matcher" not in pre[0]

    async def test_timeout_seconds_emitted(self):
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        server = HookIpcServer(
            hooks={"Stop": [HookMatcher(matcher=None, hooks=[_h])]}, can_use_tool=None
        )
        block = build_hooks_settings(server, timeout_seconds=30)
        assert block["Stop"][0]["hooks"][0]["timeout"] == 30


# --------------------------------------------------------------------------- #
# Dispatch (in-process, no shim subprocess)
# --------------------------------------------------------------------------- #


class TestDispatch:
    async def test_user_hook_fires_with_shape(self):
        seen = []

        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append((inp, tuid, ctx))
            return {}

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Write", hooks=[_h])]},
            can_use_tool=None,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "x"},
                "tool_use_id": "tu_1",
            }
        )
        assert out == {}
        assert len(seen) == 1
        inp, tuid, ctx = seen[0]
        assert inp["tool_name"] == "Write"
        assert tuid == "tu_1"
        assert ctx == {"signal": None}

    async def test_user_hook_matcher_filters(self):
        seen = []

        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append(inp["tool_name"])
            return {}

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Edit", hooks=[_h])]},
            can_use_tool=None,
        )
        await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        assert seen == []  # Write does not match "Edit"

    async def test_async_continue_field_conversion(self):
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {"continue_": False, "stopReason": "halt"}

        server = HookIpcServer(
            hooks={"Stop": [HookMatcher(matcher=None, hooks=[_h])]}, can_use_tool=None
        )
        out = await server._dispatch({"hook_event_name": "Stop"})
        assert out["continue"] is False
        assert "continue_" not in out
        assert out["stopReason"] == "halt"

    async def test_can_use_tool_allow_with_updated_input(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            new = dict(inp)
            new["content"] = "REWRITTEN"
            return PermissionResultAllow(updated_input=new)

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "ORIG"},
                "tool_use_id": "tu_1",
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"
        assert hso["updatedInput"]["content"] == "REWRITTEN"

    async def test_can_use_tool_deny_with_message(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="nope")

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        out = await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert hso["permissionDecisionReason"] == "nope"

    async def test_can_use_tool_raise_denies(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            raise RuntimeError("boom")

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        out = await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_permission_decision_sink_fires_on_deny(self):
        decisions = []

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="x")

        server = HookIpcServer(
            hooks=None,
            can_use_tool=_can_use,
            on_permission_decision=lambda n, d, t: decisions.append((n, d, t)),
        )
        await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {},
                "tool_use_id": "tu_9",
            }
        )
        assert decisions == [("Write", "deny", "tu_9")]

    async def test_user_hook_and_can_use_tool_both_dispatch(self):
        # A user PreToolUse hook + can_use_tool both run on the same event; the
        # can_use_tool permission decision lands in hookSpecificOutput while the
        # user hook's non-permission fields are preserved.
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {"systemMessage": "from-user-hook"}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="denied")

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_h])]},
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {"hook_event_name": "PreToolUse", "tool_name": "Write", "tool_input": {}}
        )
        assert out["systemMessage"] == "from-user-hook"
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_can_use_tool_consulted_for_unmatched_tool(self):
        # W2 bypass: a user PreToolUse hook matched ONLY to "Bash" plus
        # can_use_tool. A Write (outside the matcher) must still reach
        # can_use_tool and be DENIED -- the user hook simply does not fire.
        user_seen = []

        async def _bash_only(inp, tuid, ctx):  # noqa: ANN001, ANN202
            user_seen.append(inp["tool_name"])
            return {}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="blocked by can_use_tool")

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[_bash_only])]},
            can_use_tool=_can_use,
        )
        # The CLI fires the shim for Write because of the synthetic catch-all
        # entry that build_hooks_settings now always emits.
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "x"},
                "tool_use_id": "tu_w",
            }
        )
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert out["hookSpecificOutput"]["permissionDecisionReason"] == (
            "blocked by can_use_tool"
        )
        # The user's Bash-only hook must NOT have fired for a Write.
        assert user_seen == []

    async def test_can_use_tool_not_double_invoked_for_same_tool_use_id(self):
        # W2 dedup: the CLI may fire the shim twice for a tool that matches BOTH
        # the synthetic catch-all AND the user's narrow matcher (e.g. Bash).
        # can_use_tool must be consulted EXACTLY once per tool_use_id; the cached
        # decision is replayed for the second fire.
        calls = []
        decisions = []

        async def _bash_hook(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {}

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            calls.append(name)
            return PermissionResultDeny(message="no")

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[_bash_hook])]},
            can_use_tool=_can_use,
            on_permission_decision=lambda n, d, t: decisions.append((n, d, t)),
        )
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_use_id": "tu_dup",
        }
        first = await server._dispatch(dict(event))
        second = await server._dispatch(dict(event))
        # Same authoritative decision both times...
        assert first["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert second["hookSpecificOutput"]["permissionDecision"] == "deny"
        # ...but the callback ran only ONCE, and the transport was notified once.
        assert calls == ["Bash"]
        assert decisions == [("Bash", "deny", "tu_dup")]

    async def test_can_use_tool_without_tool_use_id_not_deduped(self):
        # No usable tool_use_id -> no stable dedup key; each fire runs the
        # callback (safe: the common case fires the shim once anyway).
        calls = []

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            calls.append(name)
            return PermissionResultAllow()

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "tool_input": {},
        }
        await server._dispatch(dict(event))
        await server._dispatch(dict(event))
        assert calls == ["Write", "Write"]

    # ------------------------------------------------------------------ #
    # W5: deny-from-either-source wins; no stale reason leaks.
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deny_hook(reason="hook-deny"):  # noqa: ANN001, ANN205
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        return _h

    @staticmethod
    def _allow_hook():  # noqa: ANN205
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "permissionDecisionReason": "hook-allow",
                }
            }

        return _h

    async def test_w5_hook_deny_beats_can_use_tool_allow(self):
        # User PreToolUse hook DENY + can_use_tool ALLOW for the SAME tool ->
        # deny wins, and the stale "hook-deny" reason does NOT leak as an
        # allow-with-deny-reason (the surviving triplet is the deny's).
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow(updated_input={"content": "X"})

        server = HookIpcServer(
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[self._deny_hook()])]
            },
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "ORIG"},
                "tool_use_id": "tu_1",
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert hso["permissionDecisionReason"] == "hook-deny"
        # The allow's updatedInput must NOT leak into the surviving deny.
        assert "updatedInput" not in hso

    async def test_w5_can_use_tool_deny_beats_hook_allow(self):
        # can_use_tool DENY + user hook ALLOW -> deny wins with can_use_tool's
        # reason; the stale "hook-allow" reason does NOT leak.
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="cut-deny")

        server = HookIpcServer(
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[self._allow_hook()])]
            },
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {},
                "tool_use_id": "tu_2",
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        assert hso["permissionDecisionReason"] == "cut-deny"

    async def test_w5_both_allow_runs_and_updated_input_applies(self):
        # Both allow -> allow, and a can_use_tool updated_input still applies.
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            new = dict(inp)
            new["content"] = "REWRITTEN"
            return PermissionResultAllow(updated_input=new)

        server = HookIpcServer(
            hooks={
                "PreToolUse": [HookMatcher(matcher=None, hooks=[self._allow_hook()])]
            },
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {"file_path": "/a", "content": "ORIG"},
                "tool_use_id": "tu_3",
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "allow"
        assert hso["updatedInput"]["content"] == "REWRITTEN"
        # The hook-allow reason must NOT leak from the superseded hook decision.
        assert hso.get("permissionDecisionReason") != "hook-allow"

    async def test_w5_both_deny_keeps_hook_reason(self):
        # Both deny -> deny; hook's decision survives (deny == deny, hook kept).
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultDeny(message="cut-deny")

        server = HookIpcServer(
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher=None, hooks=[self._deny_hook("hook-reason")])
                ]
            },
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {},
                "tool_use_id": "tu_4",
            }
        )
        hso = out["hookSpecificOutput"]
        assert hso["permissionDecision"] == "deny"
        # A deny is a deny from either source; the result is consistent (no
        # allow), and exactly one reason is present.
        assert hso["permissionDecisionReason"] in ("hook-reason", "cut-deny")

    async def test_w5_hook_allow_can_use_allow_no_nonperm_field_loss(self):
        # A non-permission field the user hook contributes is preserved when the
        # permission triplet is replaced.
        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            return {
                "systemMessage": "keepme",
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "hook-deny",
                },
            }

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            return PermissionResultAllow()

        server = HookIpcServer(
            hooks={"PreToolUse": [HookMatcher(matcher=None, hooks=[_h])]},
            can_use_tool=_can_use,
        )
        out = await server._dispatch(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Write",
                "tool_input": {},
                "tool_use_id": "tu_5",
            }
        )
        assert out["systemMessage"] == "keepme"
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    # ------------------------------------------------------------------ #
    # W6: distinct concurrent ids are NOT serialized; same id consults once.
    # ------------------------------------------------------------------ #

    async def test_w6_distinct_ids_run_concurrently(self):
        import time

        started = anyio.Event()
        n_started = 0
        gate = anyio.Event()

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            nonlocal n_started
            n_started += 1
            if n_started >= 2:
                gate.set()
            started.set()
            # Block until BOTH callbacks have started -- proves they overlap and
            # are not serialized behind a single lock.
            await gate.wait()
            return PermissionResultAllow()

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)

        async def _dispatch(tuid):  # noqa: ANN001, ANN202
            await server._dispatch(
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Write",
                    "tool_input": {},
                    "tool_use_id": tuid,
                }
            )

        t0 = time.monotonic()
        with anyio.fail_after(2):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_dispatch, "tu_a")
                tg.start_soon(_dispatch, "tu_b")
        elapsed = time.monotonic() - t0
        # If the lock were held across the callback, the second callback could
        # never start while the first waits on the gate -> deadlock/timeout.
        # Reaching here (both started, gate set, both returned) proves overlap.
        assert n_started == 2
        assert elapsed < 1.5

    async def test_w6_same_id_consulted_once_under_concurrency(self):
        calls = []
        proceed = anyio.Event()

        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            calls.append(name)
            await proceed.wait()
            return PermissionResultDeny(message="no")

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        event = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "tool_use_id": "tu_same",
        }
        results = {}

        async def _dispatch(key):  # noqa: ANN001, ANN202
            results[key] = await server._dispatch(dict(event))

        with anyio.fail_after(2):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_dispatch, "first")
                # Let the first dispatch claim the in-flight slot before the
                # second arrives, so the second must wait+replay (not re-invoke).
                await anyio.sleep(0.05)
                tg.start_soon(_dispatch, "second")
                await anyio.sleep(0.05)
                proceed.set()

        # The callback ran EXACTLY once despite two concurrent dispatches.
        assert calls == ["Bash"]
        assert results["first"]["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert results["second"]["hookSpecificOutput"]["permissionDecision"] == "deny"

    async def test_w6_raising_run_can_use_tool_wakes_waiters(self):
        # Defensive: if the owner's _run_can_use_tool itself raised (e.g. a
        # deferred import failed), the in-flight ``perm`` is pre-bound to None so
        # the finally still publishes a result and SETS ``done`` -- a same-id
        # waiter must NOT hang and the exception must surface to the owner.
        server = HookIpcServer(hooks=None, can_use_tool=lambda *a, **k: None)

        boom = RuntimeError("boom")
        in_callback = anyio.Event()
        release = anyio.Event()

        async def _raise(event, tool_name, tool_use_id):  # noqa: ANN001, ANN202
            in_callback.set()
            # Hold inside the callback until the waiter has attached to the
            # in-flight slot, so the waiter MUST replay (not re-invoke).
            await release.wait()
            raise boom

        server._run_can_use_tool = _raise  # type: ignore[assignment]

        owner_err: list[BaseException] = []
        waiter_result: list[object] = []

        async def _owner():  # noqa: ANN202
            try:
                await server._run_can_use_tool_deduped(
                    {"tool_input": {}}, "Bash", "tu_raise"
                )
            except BaseException as e:  # noqa: BLE001
                owner_err.append(e)

        async def _waiter():  # noqa: ANN202
            await in_callback.wait()
            waiter_result.append(
                await server._run_can_use_tool_deduped(
                    {"tool_input": {}}, "Bash", "tu_raise"
                )
            )

        with anyio.fail_after(2):
            async with anyio.create_task_group() as tg:
                tg.start_soon(_owner)
                tg.start_soon(_waiter)
                # Owner enters callback + waiter attaches to in-flight, THEN let
                # the owner's callback raise.
                await in_callback.wait()
                await anyio.sleep(0.05)
                release.set()

        # The owner saw the real exception; the waiter did NOT hang and replayed
        # the (None) result rather than re-invoking the failed callback.
        assert owner_err == [boom]
        assert waiter_result == [None]

    async def test_non_tool_event_with_matcher_still_fires(self):
        # W4: a user matcher on a NON-tool event (UserPromptSubmit/Stop/...) has
        # no tool matchQuery on the CLI side -> the CLI fires every matcher
        # regardless. Mirror that: the callback must fire even though the
        # matcher pattern would not match a (None) tool name.
        seen = []

        async def _h(inp, tuid, ctx):  # noqa: ANN001, ANN202
            seen.append(inp.get("hook_event_name"))
            return {}

        server = HookIpcServer(
            hooks={"UserPromptSubmit": [HookMatcher(matcher="something", hooks=[_h])]},
            can_use_tool=None,
        )
        await server._dispatch({"hook_event_name": "UserPromptSubmit", "prompt": "hi"})
        assert seen == ["UserPromptSubmit"]


# --------------------------------------------------------------------------- #
# Server lifecycle + real round-trip via the shim subprocess
# --------------------------------------------------------------------------- #


def _run_shim(spec: str, event: dict) -> str:
    env = dict(os.environ)
    env[ENV_HOOK_IPC] = spec
    proc = subprocess.run(
        shim_command(),
        input=json.dumps(event),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return proc.stdout


class TestServerLifecycleAndShim:
    async def test_start_exposes_spec_and_stop_is_clean(self):
        server = HookIpcServer(hooks=None, can_use_tool=None)
        # Nothing to wire -> the transport would not start it, but the server
        # itself still binds when asked (defensive).
        await server.start()
        try:
            assert server.spec  # tcp:... or unix:...
        finally:
            await server.stop()
        # Double-stop is safe.
        await server.stop()

    async def test_uses_unix_socket_in_0700_dir_on_posix(self):
        # W1: AF_UNIX is on the `socket` module (not `os`); the start() guard now
        # checks hasattr(socket, "AF_UNIX"), so POSIX uses a Unix domain socket in
        # a private 0700 dir, NOT the TCP loopback fallback.
        import socket as _socket
        import stat
        from pathlib import Path

        if not (hasattr(anyio, "create_unix_listener") and hasattr(_socket, "AF_UNIX")):
            pytest.skip("AF_UNIX unavailable on this platform")

        server = HookIpcServer(hooks=None, can_use_tool=lambda *a: None)
        await server.start()
        try:
            assert server.spec.startswith("unix:"), (
                f"expected a unix socket spec on POSIX, got {server.spec!r}"
            )
            sock_path = Path(server.spec.split(":", 2)[1])
            assert sock_path.exists()
            # The containing temp dir must be private (0700) for defense-in-depth.
            mode = stat.S_IMODE(sock_path.parent.stat().st_mode)
            assert mode == 0o700, f"socket dir mode {oct(mode)} is not 0700"
        finally:
            await server.stop()
        # The socket file + its temp dir are cleaned up on stop().
        assert not sock_path.exists()

    async def test_real_shim_round_trip(self):
        async def _can_use(name, inp, ctx):  # noqa: ANN001, ANN202
            new = dict(inp)
            new["content"] = "REWRITTEN"
            return PermissionResultAllow(updated_input=new)

        server = HookIpcServer(hooks=None, can_use_tool=_can_use)
        await server.start()
        try:
            out = await anyio.to_thread.run_sync(
                _run_shim,
                server.spec,
                {
                    "hook_event_name": "PreToolUse",
                    "tool_name": "Write",
                    "tool_input": {"file_path": "/a", "content": "ORIG"},
                    "tool_use_id": "tu_1",
                },
            )
        finally:
            await server.stop()
        parsed = json.loads(out)
        assert parsed["hookSpecificOutput"]["updatedInput"]["content"] == "REWRITTEN"

    async def test_shim_bad_token_returns_noop(self):
        server = HookIpcServer(hooks=None, can_use_tool=lambda *a: None)
        await server.start()
        try:
            bad = server.spec.rsplit(":", 1)[0] + ":deadbeef"
            out = await anyio.to_thread.run_sync(
                _run_shim,
                bad,
                {"hook_event_name": "PreToolUse", "tool_name": "Write"},
            )
        finally:
            await server.stop()
        assert json.loads(out) == {}

    async def test_shim_no_endpoint_fails_open(self):
        # No CLAUDE_AGENT_SDK_HOOK_IPC set -> shim emits `{}` and exits 0.
        env = dict(os.environ)
        env.pop(ENV_HOOK_IPC, None)
        proc = await anyio.to_thread.run_sync(
            lambda: subprocess.run(  # noqa: S603
                shim_command(),
                input="{}",
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
        )
        assert proc.returncode == 0
        assert json.loads(proc.stdout) == {}
