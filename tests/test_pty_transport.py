"""Tests for the PTY-based interactive CLI transport.

These cover the transport's logic without a real CLI: transcript translation,
command construction, env handling, keystroke encoding, the transcript tail
loop, and the control-request -> interactive-keystroke mappings. A full
end-to-end run against a fake ``claude`` lives in ``test_pty_integration.py``.
"""

import json
from unittest.mock import patch

import anyio
import pytest

from claude_agent_sdk._internal.transport import pty_cli
from claude_agent_sdk._internal.transport.pty_cli import (
    _SHIFT_TAB,
    PtyCLITransport,
    _sanitize_assistant_message,
    _translate_transcript_entry,
)
from claude_agent_sdk.types import ClaudeAgentOptions

DEFAULT_CLI = "/usr/bin/claude"


def make_transport(**kwargs: object) -> PtyCLITransport:
    cli_path = kwargs.pop("cli_path", DEFAULT_CLI)
    options = ClaudeAgentOptions(cli_path=cli_path, **kwargs)
    return PtyCLITransport(prompt="hi", options=options)


# --------------------------------------------------------------------------- #
# Transcript translation
# --------------------------------------------------------------------------- #


class TestTranslateTranscriptEntry:
    def test_assistant_entry_passes_message_through(self):
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "pong"}],
            },
            "sessionId": "s-1",
            "uuid": "u-1",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["type"] == "assistant"
        assert out["session_id"] == "s-1"
        assert out["uuid"] == "u-1"
        assert out["message"]["content"][0]["text"] == "pong"

    def test_user_plain_text_prompt_is_skipped(self):
        # Plain-text user records are the echo of what we typed; not surfaced.
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "reply with pong"},
            "sessionId": "s-1",
        }
        assert _translate_transcript_entry(entry, "fallback") is None

    def test_user_tool_result_is_surfaced(self):
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
                ],
            },
            "sessionId": "s-1",
            "uuid": "u-2",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["type"] == "user"
        assert out["message"]["content"][0]["type"] == "tool_result"

    def test_user_list_content_without_tool_result_is_surfaced(self):
        # M4: structured (list) user records are surfaced even without a
        # tool_result block -- e.g. image/document content. Only the plain-text
        # echo of the prompt we typed (string content) is suppressed.
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            },
            "sessionId": "s-1",
            "uuid": "u-3",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["type"] == "user"
        assert out["message"]["content"][0]["type"] == "text"

    def test_user_plain_string_echo_is_skipped(self):
        # The prompt we typed is recorded as string content -> suppressed so we
        # don't double-emit it to consumers.
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "the prompt I typed"},
        }
        assert _translate_transcript_entry(entry, "fallback") is None

    def test_turn_duration_not_translated_directly(self):
        # turn_duration is handled statefully by the tail loop (_emit_result),
        # not by the pure translate function.
        entry = {"type": "system", "subtype": "turn_duration", "durationMs": 1234}
        assert _translate_transcript_entry(entry, "fallback") is None

    def test_assistant_string_content_coerced(self):
        # The transcript may write content as a bare string; it must not crash
        # the parser (which iterates content blocks).
        entry = {
            "type": "assistant",
            "message": {"role": "assistant", "content": "hi there"},
            "sessionId": "s",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["message"]["content"] == [{"type": "text", "text": "hi there"}]
        assert out["message"]["model"] == "unknown"  # backfilled

    def test_parent_tool_use_id_carried(self):
        entry = {
            "type": "assistant",
            "message": {"role": "assistant", "model": "m", "content": []},
            "parentToolUseId": "tool-7",
        }
        out = _translate_transcript_entry(entry, "fallback")
        assert out is not None
        assert out["parent_tool_use_id"] == "tool-7"

    @pytest.mark.parametrize(
        "entry_type",
        ["queue-operation", "last-prompt", "ai-title", "attachment", "mode"],
    )
    def test_bookkeeping_types_translate_to_none(self, entry_type):
        assert _translate_transcript_entry({"type": entry_type}, "fallback") is None

    def test_falls_back_to_default_session_id(self):
        entry = {
            "type": "assistant",
            "message": {"role": "assistant", "model": "m", "content": []},
        }
        out = _translate_transcript_entry(entry, "fallback-sid")
        assert out is not None
        assert out["session_id"] == "fallback-sid"


class TestSanitizeAssistantMessage:
    def test_backfills_missing_thinking_signature(self):
        message = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "hmm"}],
        }
        out = _sanitize_assistant_message(message)
        assert out["content"][0]["signature"] == ""

    def test_preserves_existing_signature(self):
        message = {
            "content": [{"type": "thinking", "thinking": "x", "signature": "sig"}],
        }
        out = _sanitize_assistant_message(message)
        assert out["content"][0]["signature"] == "sig"


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


class TestBuildCommand:
    def test_strips_stream_json_io_flags(self):
        t = make_transport()
        t._cli_path = DEFAULT_CLI
        cmd = t._build_command()
        # Interactive mode must not carry the headless stream-json protocol.
        assert "--input-format" not in cmd
        assert "--output-format" not in cmd
        assert "--verbose" not in cmd
        assert "stream-json" not in cmd
        assert "--print" not in cmd
        assert "-p" not in cmd

    def test_includes_session_id_for_transcript_discovery(self):
        t = make_transport()
        t._cli_path = DEFAULT_CLI
        cmd = t._build_command()
        assert "--session-id" in cmd
        assert cmd[cmd.index("--session-id") + 1] == t._session_id

    def test_forwards_common_options(self):
        t = make_transport(model="claude-sonnet-4-6", permission_mode="acceptEdits")
        t._cli_path = DEFAULT_CLI
        cmd = t._build_command()
        assert cmd[cmd.index("--model") + 1] == "claude-sonnet-4-6"
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"


class TestComputeTranscriptPath:
    def test_path_ends_with_session_jsonl(self):
        t = make_transport(session_id="11111111-1111-1111-1111-111111111111")
        path = t._compute_transcript_path()
        assert path.name == "11111111-1111-1111-1111-111111111111.jsonl"
        assert "projects" in path.parts


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #


class TestBuildEnv:
    def test_sets_pty_entrypoint_and_filters_claudecode(self):
        with patch.dict("os.environ", {"CLAUDECODE": "1"}):
            env = make_transport()._build_env()
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"
        assert "CLAUDECODE" not in env
        assert "CLAUDE_AGENT_SDK_VERSION" in env

    def test_normalizes_is_sandbox_to_one_when_root(self):
        with (
            patch("os.geteuid", return_value=0),
            patch.dict("os.environ", {"IS_SANDBOX": "yes"}),
        ):
            env = make_transport()._build_env()
        # The CLI only accepts the exact value "1".
        assert env["IS_SANDBOX"] == "1"

    def test_caller_can_override_is_sandbox(self):
        with patch("os.geteuid", return_value=0):
            env = make_transport(env={"IS_SANDBOX": "custom"})._build_env()
        assert env["IS_SANDBOX"] == "custom"


# --------------------------------------------------------------------------- #
# Transport interface behavior (no real subprocess)
# --------------------------------------------------------------------------- #


class TestRecoverToolInputRace:
    """RV2: the permission path bounded-awaits the relay tee before falling back.

    Reproduces the race the reviewer found (dialog detected before the tee
    populated the recovery map) and asserts the bounded await resolves it
    deterministically instead of handing can_use_tool the scraped {target}.
    """

    def test_await_resolves_after_tee_lands(self):
        async def _test():
            t = make_transport()
            # A monitor must be present for the await to engage (else it would be
            # pointless to wait -- the maps would never fill).
            t._api_monitor = object()  # type: ignore[assignment]

            async with anyio.create_task_group() as tg:
                # Simulate the tee landing ~120ms AFTER the dialog is detected.
                async def _populate_later() -> None:
                    await anyio.sleep(0.12)
                    entry = {
                        "name": "Write",
                        "input": {"file_path": "/tmp/rv2.txt", "content": "x"},
                    }
                    t._turn_tool_inputs["toolu_real"] = entry
                    t._turn_tool_input_by_name["Write"] = entry

                tg.start_soon(_populate_later)
                # At call time the maps are EMPTY (the race window).
                assert t._recover_tool_input("Write") == (None, None)
                full_input, tool_use_id = await t._await_recovered_tool_input(
                    "Write", timeout_s=2.0, poll_s=0.01
                )

            assert full_input == {"file_path": "/tmp/rv2.txt", "content": "x"}
            assert tool_use_id == "toolu_real"

        anyio.run(_test)

    def test_await_times_out_to_none_when_never_populated(self):
        async def _test():
            t = make_transport()
            t._api_monitor = object()  # type: ignore[assignment]
            full_input, tool_use_id = await t._await_recovered_tool_input(
                "Write", timeout_s=0.1, poll_s=0.01
            )
            assert full_input is None
            assert tool_use_id is None

        anyio.run(_test)

    def test_no_monitor_does_not_wait(self):
        async def _test():
            t = make_transport()
            t._api_monitor = None
            # With no monitor there is nothing to await: returns immediately.
            import time as _time

            start = _time.monotonic()
            result = await t._await_recovered_tool_input(
                "Write", timeout_s=5.0, poll_s=0.01
            )
            assert result == (None, None)
            assert _time.monotonic() - start < 0.5

        anyio.run(_test)


class TestWriteRouting:
    def test_control_request_enqueues_success_response(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "req-7",
                        "request": {"subtype": "initialize"},
                    }
                )
                + "\n"
            )
            msg = t._out_recv.receive_nowait()
            assert msg["type"] == "control_response"
            assert msg["response"]["subtype"] == "success"
            assert msg["response"]["request_id"] == "req-7"

        anyio.run(_test)

    def test_user_message_is_typed_into_pty(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            typed: list[str] = []

            async def fake_type(text: str) -> None:
                typed.append(text)

            t._type_prompt = fake_type  # type: ignore[method-assign]
            await t.write(
                json.dumps(
                    {
                        "type": "user",
                        "message": {"role": "user", "content": "hello there"},
                    }
                )
                + "\n"
            )
            assert typed == ["hello there"]

        anyio.run(_test)

    def test_interrupt_writes_escape(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            written: list[bytes] = []

            async def fake_pty_write(data: bytes) -> None:
                written.append(data)

            t._pty_write = fake_pty_write  # type: ignore[method-assign]
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "r",
                        "request": {"subtype": "interrupt"},
                    }
                )
                + "\n"
            )
            assert b"\x1b" in written

        anyio.run(_test)


class TestEndInput:
    def test_end_input_sets_flag(self):
        async def _test():
            t = make_transport()
            assert t._input_ended is False
            await t.end_input()
            assert t._input_ended is True

        anyio.run(_test)


class TestTypePrompt:
    def test_newlines_preserved_via_bracketed_paste(self):
        async def _test():
            t = make_transport()
            t._warmed_up = True  # skip the startup-toast warmup delay
            writes: list[bytes] = []

            async def fake_pty_write(data: bytes) -> None:
                writes.append(data)

            t._pty_write = fake_pty_write  # type: ignore[method-assign]
            await t._type_prompt("line one\nline two")
            joined = b"".join(writes)
            # Bracketed-paste wrapped, newline preserved (not collapsed), then CR.
            assert b"\x1b[200~line one\nline two\x1b[201~" in joined
            assert writes[-1] == b"\r"

        anyio.run(_test)

    @pytest.mark.parametrize("lead", ["/", "!", "#"])
    def test_leading_tui_reserved_char_gets_space_prefix(self, lead):
        async def _test():
            t = make_transport()
            t._warmed_up = True
            writes: list[bytes] = []

            async def fake_pty_write(data: bytes) -> None:
                writes.append(data)

            t._pty_write = fake_pty_write  # type: ignore[method-assign]
            await t._type_prompt(f"{lead}do something")
            joined = b"".join(writes)
            # A single leading space is inserted so the TUI does not enter
            # slash/bash/memory command mode.
            assert f"\x1b[200~ {lead}do something\x1b[201~".encode() in joined

        anyio.run(_test)

    def test_normal_prompt_not_prefixed(self):
        async def _test():
            t = make_transport()
            t._warmed_up = True
            writes: list[bytes] = []

            async def fake_pty_write(data: bytes) -> None:
                writes.append(data)

            t._pty_write = fake_pty_write  # type: ignore[method-assign]
            await t._type_prompt("hello")
            assert b"\x1b[200~hello\x1b[201~" in b"".join(writes)

        anyio.run(_test)


# --------------------------------------------------------------------------- #
# Transcript tail loop (no subprocess) — feed a file, collect SDK messages
# --------------------------------------------------------------------------- #


def _drain(transport: PtyCLITransport) -> list[dict]:
    out: list[dict] = []
    while True:
        try:
            out.append(transport._out_recv.receive_nowait())
        except anyio.WouldBlock:
            break
        except anyio.EndOfStream:
            break
    return out


class TestTailLoop:
    def _run_with_transcript(self, tmp_path, lines: list[dict]) -> list[dict]:
        async def _test() -> list[dict]:
            t = make_transport()
            path = tmp_path / "sess.jsonl"
            path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # One-shot: stop once a result is emitted.
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        return anyio.run(_test)

    def test_assistant_and_result_emitted(self, tmp_path):
        msgs = self._run_with_transcript(
            tmp_path,
            [
                {
                    "type": "user",
                    "sessionId": "s",
                    "message": {"role": "user", "content": "hi"},
                },
                {
                    "type": "assistant",
                    "sessionId": "s",
                    "uuid": "a1",
                    "message": {
                        "role": "assistant",
                        "model": "m",
                        "content": [{"type": "text", "text": "yo"}],
                    },
                },
                {
                    "type": "system",
                    "subtype": "turn_duration",
                    "sessionId": "s",
                    "durationMs": 3,
                },
            ],
        )
        types = [m["type"] for m in msgs]
        # plain-text user record is dropped; assistant + synthesized result remain
        assert types == ["assistant", "result"]
        assert msgs[0]["message"]["content"][0]["text"] == "yo"
        assert msgs[1]["subtype"] == "success"

    def test_skips_bookkeeping_records(self, tmp_path):
        msgs = self._run_with_transcript(
            tmp_path,
            [
                {"type": "queue-operation", "sessionId": "s"},
                {"type": "file-history-snapshot", "sessionId": "s"},
                {
                    "type": "assistant",
                    "sessionId": "s",
                    "uuid": "a1",
                    "message": {
                        "role": "assistant",
                        "model": "m",
                        "content": [{"type": "text", "text": "ok"}],
                    },
                },
                {"type": "system", "subtype": "turn_duration", "sessionId": "s"},
            ],
        )
        assert [m["type"] for m in msgs] == ["assistant", "result"]

    def test_tool_result_user_record_surfaces(self, tmp_path):
        msgs = self._run_with_transcript(
            tmp_path,
            [
                {
                    "type": "user",
                    "sessionId": "s",
                    "uuid": "u1",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "t1",
                                "content": "out",
                            }
                        ],
                    },
                },
                {"type": "system", "subtype": "turn_duration", "sessionId": "s"},
            ],
        )
        assert msgs[0]["type"] == "user"
        assert msgs[0]["message"]["content"][0]["type"] == "tool_result"

    def test_ignores_malformed_lines(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "sess.jsonl"
            path.write_text(
                "not json\n"
                + json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": "s",
                        "uuid": "a",
                        "message": {
                            "role": "assistant",
                            "model": "m",
                            "content": [{"type": "text", "text": "x"}],
                        },
                    }
                )
                + "\n"
                + json.dumps({"type": "system", "subtype": "turn_duration"})
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        assert [m["type"] for m in msgs] == ["assistant", "result"]


class TestEarlyExit:
    def test_nonzero_exit_emits_error_result(self):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._recent_output = b"\x1b[31mfatal: boom\x1b[0m\n"
            await t._handle_early_exit(2)
            return _drain(t)

        msgs = anyio.run(_test)
        assert len(msgs) == 1
        assert msgs[0]["type"] == "result"
        assert msgs[0]["is_error"] is True
        assert "boom" in msgs[0]["result"]

    def test_clean_exit_emits_nothing(self):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_early_exit(0)
            return _drain(t)

        assert anyio.run(_test) == []


# --------------------------------------------------------------------------- #
# Control-request -> interactive keystroke/slash-command mappings
# --------------------------------------------------------------------------- #


def _capture_pty_writes(t: PtyCLITransport) -> list[bytes]:
    writes: list[bytes] = []

    async def fake_pty_write(data: bytes) -> None:
        writes.append(data)

    t._pty_write = fake_pty_write  # type: ignore[method-assign]
    t._warmed_up = True
    return writes


class TestControlMappings:
    @pytest.mark.parametrize(
        ("start", "target", "expected_shift_tabs"),
        [
            ("default", "acceptEdits", 1),
            ("default", "plan", 2),
            ("plan", "default", 1),  # wraps around the 3-cycle
            ("acceptEdits", "default", 2),
        ],
    )
    def test_set_permission_mode_cycles(self, start, target, expected_shift_tabs):
        async def _test():
            t = make_transport()
            t._permission_mode = start
            writes = _capture_pty_writes(t)
            await t._set_permission_mode(target)
            assert writes.count(_SHIFT_TAB) == expected_shift_tabs
            assert t._permission_mode == target

        anyio.run(_test)

    def test_set_permission_mode_noop_when_same(self):
        async def _test():
            t = make_transport()
            t._permission_mode = "plan"
            writes = _capture_pty_writes(t)
            await t._set_permission_mode("plan")
            assert writes == []

        anyio.run(_test)

    def test_set_permission_mode_bypass_not_cyclable(self):
        async def _test():
            t = make_transport()
            t._permission_mode = "default"
            writes = _capture_pty_writes(t)
            await t._set_permission_mode("bypassPermissions")
            assert _SHIFT_TAB not in writes  # cannot reach by cycling
            assert t._permission_mode == "default"

        anyio.run(_test)

    def test_permission_mode_record_tracks_live_mode(self, tmp_path):
        # The CLI's permission-mode transcript record is the source of truth and
        # corrects drift (L5) / confirms set_permission_mode (H1).
        async def _test():
            t = make_transport()
            t._permission_mode = "default"
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._emit_line(
                json.dumps(
                    {"type": "permission-mode", "permissionMode": "plan"}
                ).encode()
            )
            return t

        t = anyio.run(_test)
        assert t._permission_mode == "plan"
        # And it is not surfaced as an SDK message (bookkeeping only).
        assert _drain(t) == []

    def test_set_model_types_slash_model(self):
        async def _test():
            t = make_transport()
            writes = _capture_pty_writes(t)
            await t._set_model("opus")
            joined = b"".join(writes)
            assert b"/model opus" in joined
            assert writes[-1] == b"\r"

        anyio.run(_test)

    def test_set_model_none_is_noop(self):
        async def _test():
            t = make_transport()
            writes = _capture_pty_writes(t)
            await t._set_model(None)
            assert writes == []

        anyio.run(_test)

    @pytest.mark.parametrize(
        "subtype",
        ["rewind_files", "stop_task", "mcp_toggle"],
    )
    def test_unsupported_controls_return_error(self, subtype):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_control_request(
                {
                    "type": "control_request",
                    "request_id": "r",
                    "request": {"subtype": subtype},
                }
            )
            # An explicit error response (not a fake success) so callers see it.
            resp = t._out_recv.receive_nowait()
            assert resp["type"] == "control_response"
            assert resp["response"]["subtype"] == "error"
            assert "not supported" in resp["response"]["error"]

        anyio.run(_test)

    def test_initialize_returns_server_info_baseline_keyset(self):
        """initialize control response uses the get_server_info() key set (R5).

        This is the initialize CONTROL-RESPONSE shape (commands /
        available_output_styles / models / account / pid / agents /
        output_style), plus the RL10 traffic-derived tools / model. The pure
        system/init-only key ``permissionMode`` still does not leak in.
        get_server_info() consumers do info.get('commands', []) etc., so every
        baseline key must be present.
        """

        async def _test():
            t = make_transport(
                model="claude-opus-4-8",
                allowed_tools=["Read", "Write"],
                permission_mode="acceptEdits",
            )
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_control_request(
                {
                    "type": "control_request",
                    "request_id": "r",
                    "request": {"subtype": "initialize"},
                }
            )
            resp = t._out_recv.receive_nowait()
            assert resp["response"]["subtype"] == "success"
            info = resp["response"]["response"]
            # Baseline top key set (from a live old-SDK get_server_info()).
            for key in (
                "commands",
                "available_output_styles",
                "output_style",
                "models",
                "account",
                "agents",
                "pid",
            ):
                assert key in info, f"missing baseline key {key!r}"
            # Documented .get() defaults are the right container types.
            assert isinstance(info["commands"], list)
            assert isinstance(info["available_output_styles"], list)
            assert isinstance(info["models"], list)
            assert isinstance(info["account"], dict)
            assert info["output_style"] == "default"
            # RL10: the real resolved tool catalog + model ARE surfaced on
            # get_server_info (populated from the relay-teed request when seen;
            # an empty list before the first request, never fabricated).
            assert "tools" in info and isinstance(info["tools"], list)
            assert "model" in info
            # A pure system/init MESSAGE key with no server-info equivalent must
            # still NOT leak into the initialize control-response shape.
            assert "permissionMode" not in info

        anyio.run(_test)

    def test_initialize_backfills_agents_from_options(self):
        async def _test():
            from claude_agent_sdk.types import AgentDefinition

            t = make_transport(
                agents={
                    "reviewer": AgentDefinition(
                        description="d", prompt="p", tools=None, model=None
                    )
                }
            )
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_control_request(
                {
                    "type": "control_request",
                    "request_id": "r",
                    "request": {"subtype": "initialize"},
                }
            )
            info = t._out_recv.receive_nowait()["response"]["response"]
            assert info["agents"] == [{"name": "reviewer"}]

        anyio.run(_test)

    def test_system_init_message_carries_enriched_fields(self):
        """The system/init MESSAGE keeps the capability shape (R6)."""

        async def _test():
            t = make_transport(
                model="claude-opus-4-8",
                allowed_tools=["Read", "Write"],
                permission_mode="acceptEdits",
            )
            init = t._build_init_data()
            assert init["model"] == "claude-opus-4-8"
            assert init["permissionMode"] == "acceptEdits"
            assert "Read" in init["tools"] and "Write" in init["tools"]
            for key in (
                "mcp_servers",
                "slash_commands",
                "output_style",
                "cwd",
                "agents",
                "plugins",
                "skills",
            ):
                assert key in init
            # Resolved model is backfilled from the first observed assistant
            # record when the caller didn't pass one.
            t2 = make_transport()
            t2._turn_model = "claude-opus-4-8"
            assert t2._build_init_data()["model"] == "claude-opus-4-8"

        anyio.run(_test)

    def test_mcp_status_reports_configured_servers(self):
        async def _test():
            t = make_transport(
                mcp_servers={
                    "fs": {"type": "stdio", "command": "x"},
                }
            )
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_control_request(
                {
                    "type": "control_request",
                    "request_id": "r",
                    "request": {"subtype": "mcp_status"},
                }
            )
            resp = t._out_recv.receive_nowait()
            assert resp["response"]["subtype"] == "success"
            servers = resp["response"]["response"]["mcpServers"]
            assert servers[0]["name"] == "fs"
            assert servers[0]["status"] == "pending"

        anyio.run(_test)

    def test_set_permission_mode_uncyclable_returns_error_response(self):
        async def _test():
            t = make_transport()
            t._permission_mode = "default"
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            await t._handle_control_request(
                {
                    "type": "control_request",
                    "request_id": "r",
                    "request": {
                        "subtype": "set_permission_mode",
                        "mode": "bypassPermissions",
                    },
                }
            )
            resp = t._out_recv.receive_nowait()
            assert resp["response"]["subtype"] == "error"
            assert "cannot be set live" in resp["response"]["error"]

        anyio.run(_test)


class TestReadMessageOrdering:
    def test_init_then_control_response(self):
        """connect() enqueues the init message; the initialize handshake then
        gets a control_response, so consumers see init first."""

        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            # Simulate what connect() emits, then the initialize handshake.
            await t._out_send.send(
                {"type": "system", "subtype": "init", "session_id": t._session_id}
            )
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "init-1",
                        "request": {"subtype": "initialize"},
                    }
                )
                + "\n"
            )
            first = t._out_recv.receive_nowait()
            second = t._out_recv.receive_nowait()
            assert first["type"] == "system" and first["subtype"] == "init"
            assert second["type"] == "control_response"
            assert second["response"]["request_id"] == "init-1"

        anyio.run(_test)


class TestPermissionAnswering:
    """C5/C6: answering blocking TUI permission dialogs."""

    @staticmethod
    def _permission_question(*, deny_opt=True):
        from claude_agent_sdk._internal.transport.pty_question import (
            DetectedQuestion,
            QuestionOption,
        )

        opts = [QuestionOption(index=1, label="Yes", action="allow_once")]
        if deny_opt:
            opts.append(QuestionOption(index=2, label="No", action="deny"))
        return DetectedQuestion(
            kind="permission",
            question="Allow Write?",
            options=opts,
            tool="Write",
            target="note.txt",
        )

    def test_default_allows_to_complete_turn(self):
        # C6: with no callback, the dialog is answered "allow" (option 1) so the
        # turn does not hang.
        async def _test():
            t = make_transport()
            writes = _capture_pty_writes(t)
            answered = await t._answer_question(self._permission_question())
            assert answered is True
            joined = b"".join(writes)
            assert b"1" in joined and writes[-1] == b"\r"

        anyio.run(_test)

    def test_can_use_tool_allow(self):
        async def _test():
            from claude_agent_sdk import PermissionResultAllow

            async def cb(tool, tool_input, ctx):
                assert tool == "Write"
                assert tool_input == {"target": "note.txt"}
                return PermissionResultAllow()

            t = make_transport(can_use_tool=cb)
            writes = _capture_pty_writes(t)
            await t._answer_question(self._permission_question())
            joined = b"".join(writes)
            assert b"1" in joined  # allow_once option

        anyio.run(_test)

    def test_can_use_tool_deny_selects_deny_and_queues_pending(self):
        # The denial entry is no longer built from the screen-scraped target;
        # it is deferred and correlated to the rejected tool_result in the
        # transcript (RR2). Answering deny queues the tool name and marks the
        # turn deny-terminated (RR1).
        async def _test():
            from claude_agent_sdk import PermissionResultDeny

            async def cb(tool, tool_input, ctx):
                return PermissionResultDeny(message="nope")

            t = make_transport(can_use_tool=cb)
            writes = _capture_pty_writes(t)
            await t._answer_question(self._permission_question())
            joined = b"".join(writes)
            assert b"2" in joined  # deny option
            assert t._pending_denied_tools == ["Write"]
            assert t._deny_terminated is True
            # No synthetic {target:...} entry recorded at answer time.
            assert t._turn_permission_denials == []

        anyio.run(_test)

    def test_callback_exception_denies(self):
        async def _test():
            async def cb(tool, tool_input, ctx):
                raise RuntimeError("boom")

            t = make_transport(can_use_tool=cb)
            decision = await t._decide_permission(self._permission_question())
            assert decision == "deny"

        anyio.run(_test)

    def test_ask_and_app_dialogs_not_auto_answered(self):
        async def _test():
            from claude_agent_sdk._internal.transport.pty_question import (
                DetectedQuestion,
                QuestionOption,
            )

            t = make_transport()
            q = DetectedQuestion(
                kind="ask",
                question="Which?",
                options=[QuestionOption(index=1, label="A")],
            )
            assert await t._answer_question(q) is False

        anyio.run(_test)

    def test_fingerprint_stable_and_distinct(self):
        q1 = self._permission_question()
        q2 = self._permission_question()
        assert PtyCLITransport._question_fingerprint(
            q1
        ) == PtyCLITransport._question_fingerprint(q2)
        q3 = self._permission_question(deny_opt=False)
        assert PtyCLITransport._question_fingerprint(
            q1
        ) != PtyCLITransport._question_fingerprint(q3)


class TestWarmupConfigurable:
    def test_warmup_seconds_is_patchable(self, monkeypatch):
        """Integration tests rely on shrinking the warmup; guard the knob."""
        monkeypatch.setattr(pty_cli, "_WARMUP_SECONDS", 0.0)
        assert pty_cli._WARMUP_SECONDS == 0.0


# --------------------------------------------------------------------------- #
# Option validation — fail loud for unsupportable features
# --------------------------------------------------------------------------- #


class TestValidateOptions:
    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            (
                {"hooks": {"PreToolUse": [{"hooks": [lambda *a: None]}]}},
                "hooks",
            ),
            (
                {"permission_prompt_tool_name": "my_tool"},
                "permission_prompt_tool_name",
            ),
        ],
    )
    def test_unsupported_options_raise(self, kwargs, needle):
        from claude_agent_sdk._errors import CLIConnectionError

        t = make_transport(**kwargs)
        with pytest.raises(CLIConnectionError) as exc:
            t._validate_options()
        assert needle in str(exc.value)

    def test_can_use_tool_is_accepted(self):
        # can_use_tool is now answered via the TUI question detector (C5), so it
        # must not be rejected at validation.
        async def cb(*_a):
            from claude_agent_sdk import PermissionResultAllow

            return PermissionResultAllow()

        t = make_transport(can_use_tool=cb)
        t._validate_options()  # must not raise

    def test_stdio_permission_sentinel_accepted(self):
        # The client sets permission_prompt_tool_name="stdio" when can_use_tool
        # is provided; that sentinel must be accepted (handled via the detector).
        t = make_transport(permission_prompt_tool_name="stdio")
        t._validate_options()  # must not raise

    def test_sdk_mcp_server_rejected_external_allowed(self):
        from claude_agent_sdk._errors import CLIConnectionError

        # An in-process SDK server is rejected...
        t = make_transport(
            mcp_servers={"x": {"type": "sdk", "name": "x", "instance": object()}}
        )
        with pytest.raises(CLIConnectionError):
            t._validate_options()

        # ...but external (stdio/http) servers validate fine.
        t2 = make_transport(mcp_servers={"y": {"type": "stdio", "command": "srv"}})
        t2._validate_options()  # must not raise

    def test_supported_options_pass(self):
        t = make_transport(model="opus", permission_mode="acceptEdits")
        t._validate_options()  # must not raise

    def test_max_buffer_size_honored(self):
        # M8: the configured buffer size bounds the message stream buffer.
        async def _test():
            t = make_transport(max_buffer_size=7)
            # Reproduce the connect()-time buffer construction.
            size = t._options.max_buffer_size or 1000
            assert size == 7

        anyio.run(_test)

    def test_observability_flags_warn_not_raise(self, caplog):
        t = make_transport(include_hook_events=True)
        with caplog.at_level("WARNING"):
            t._validate_options()  # must not raise
        assert any("include_hook_events" in r.message for r in caplog.records)

    def test_include_partial_messages_does_not_warn(self, caplog):
        # RL9-warn: include_partial_messages IS honored (StreamEvents emitted from
        # the relay-teed SSE stream), so the stale "no effect" warning is gone.
        t = make_transport(include_partial_messages=True)
        with caplog.at_level("WARNING"):
            t._validate_options()  # must not raise
        assert not any("include_partial_messages" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Result synthesis fidelity (_emit_result / usage accumulation)
# --------------------------------------------------------------------------- #


class TestResultFidelity:
    def test_result_carries_text_usage_and_turn_count(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "model": "m",
                                "content": [{"type": "text", "text": "hello "}],
                                "usage": {"input_tokens": 10, "output_tokens": 2},
                            },
                        },
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a2",
                            "message": {
                                "role": "assistant",
                                "model": "m",
                                "content": [{"type": "text", "text": "world"}],
                                "usage": {"input_tokens": 5, "output_tokens": 3},
                            },
                        },
                        {"type": "system", "subtype": "turn_duration", "durationMs": 9},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["result"] == "world"  # last assistant text
        # num_turns = tool_results + 1; no tool results here -> 1.
        assert result["num_turns"] == 1
        # usage summed across both (anonymous, distinct) assistant messages
        assert result["usage"]["input_tokens"] == 15
        assert result["usage"]["output_tokens"] == 5
        assert result["is_error"] is False

    def test_refusal_marks_error_result(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "model": "m",
                                "content": [{"type": "text", "text": "no"}],
                                "stop_reason": "refusal",
                            },
                        },
                        {"type": "system", "subtype": "turn_duration"},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["is_error"] is True

    def test_result_carries_cost_model_usage_stop_reason(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "id": "msg_1",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": "done"}],
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 1_000_000,
                                    "output_tokens": 0,
                                },
                            },
                        },
                        # A SECOND block-record sharing the same message.id but
                        # carrying a DIFFERENT block (the real transcript shape:
                        # one block per record, same id, repeated usage). Its
                        # repeated usage must NOT double-count cost/usage.
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1b",
                            "message": {
                                "role": "assistant",
                                "id": "msg_1",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": "more"}],
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 1_000_000,
                                    "output_tokens": 0,
                                },
                            },
                        },
                        {
                            "type": "system",
                            "subtype": "turn_duration",
                            "durationMs": 100,
                            "messageCount": 5,
                        },
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        # Each distinct block-record surfaces (per-block granularity, RV1): two
        # same-id records carrying different blocks -> two assistant messages.
        assert sum(1 for m in msgs if m["type"] == "assistant") == 2
        result = next(m for m in msgs if m["type"] == "result")
        # Usage/cost is deduped by message.id (repeated per block-record), so the
        # cost stays at 5.0 (NOT 10.0) despite two records sharing msg_1.
        assert result["total_cost_usd"] == 5.0
        assert result["usage"]["input_tokens"] == 1_000_000
        assert result["stop_reason"] == "end_turn"
        # num_turns = tool_results + 1; the snapshot's messageCount (5) is
        # deliberately ignored.
        assert result["num_turns"] == 1
        # Emitted under the camelCase ``modelUsage`` wire key with camelCase
        # sub-keys so message_parser (data.get("modelUsage")) populates
        # ResultMessage.model_usage (R1). The snake_case ``model_usage`` key
        # would silently parse to None at the consumer.
        assert "model_usage" not in result
        assert result["modelUsage"]["claude-opus-4-8"]["costUSD"] == 5.0
        assert result["modelUsage"]["claude-opus-4-8"]["inputTokens"] == 1_000_000
        # permission_denials is always a list (never None), like stream-json.
        assert result["permission_denials"] == []

    def test_refusal_result_subtype(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": "no"}],
                                "stop_reason": "refusal",
                            },
                        },
                        {"type": "system", "subtype": "turn_duration"},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["is_error"] is True
        assert result["stop_reason"] == "refusal"

    def test_max_turns_error_subtype(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": "stop"}],
                                "error": "max_turns exceeded",
                            },
                        },
                        {"type": "system", "subtype": "turn_duration"},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["is_error"] is True
        assert result["subtype"] == "error_max_turns"

    def test_structured_output_parsed_when_json_schema(self, tmp_path):
        async def _test():
            t = make_transport(
                output_format={
                    "type": "json_schema",
                    "schema": {"type": "object"},
                }
            )
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": '{"answer": 42}'}],
                            },
                        },
                        {"type": "system", "subtype": "turn_duration"},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["structured_output"] == {"answer": 42}

    def test_structured_output_none_without_schema(self, tmp_path):
        async def _test():
            t = make_transport()  # no output_format
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": '{"x": 1}'}],
                            },
                        },
                        {"type": "system", "subtype": "turn_duration"},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert "structured_output" not in result

    def test_num_turns_counts_tool_results_plus_one(self, tmp_path):
        # num_turns = (tool_result user records) + 1 -- each tool result is one
        # API round-trip back to the model, plus the final answer turn. The
        # turn_duration record's messageCount counts streamed snapshots, NOT API
        # turns, so it must NOT drive num_turns. Verified against live
        # stream-json num_turns. Each turn is independent: counters reset.
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"

            def assistant(n, tag):
                return {
                    "type": "assistant",
                    "sessionId": "s",
                    "uuid": f"a{n}{tag}",
                    "message": {
                        "role": "assistant",
                        "id": f"msg_{n}{tag}",
                        "model": "m",
                        "content": [{"type": "text", "text": f"r{n}"}],
                    },
                }

            def tool_result(n, tag):
                return {
                    "type": "user",
                    "sessionId": "s",
                    "uuid": f"u{n}{tag}",
                    "message": {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": f"t{tag}"}],
                    },
                }

            def end(n):
                return {
                    "type": "system",
                    "subtype": "turn_duration",
                    "uuid": f"s{n}",
                    # Deliberately a misleading large value: must be ignored.
                    "messageCount": 99,
                }

            # Turn 1: one tool round-trip -> num_turns 2.
            # Turn 2: three tool round-trips -> num_turns 4.
            lines = [
                assistant(1, "a"),
                tool_result(1, "a"),
                assistant(1, "b"),
                end(1),
            ] + [
                assistant(2, "a"),
                tool_result(2, "a"),
                tool_result(2, "b"),
                tool_result(2, "c"),
                assistant(2, "b"),
                end(2),
            ]
            path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            for x in lines:
                await t._emit_line(json.dumps(x).encode())
                # _emit_result now no-ops if a result was already emitted for the
                # turn (deny/interrupt double-emit guard); in real usage
                # _type_prompt resets this between turns, so mirror that here.
                if x.get("subtype") == "turn_duration":
                    t._result_emitted = False
            return _drain(t)

        msgs = anyio.run(_test)
        results = [m for m in msgs if m["type"] == "result"]
        assert [r["num_turns"] for r in results] == [2, 4]


class TestDenyTermination:
    """RR1/RR2/RR5: a can_use_tool deny terminates the turn faithfully."""

    @staticmethod
    def _deny_transcript():
        # An assistant tool_use followed by the rejected (is_error) tool_result
        # the CLI writes after a deny. No turn_duration record is ever written
        # (the CLI goes idle), mirroring the live behavior.
        assistant = {
            "type": "assistant",
            "sessionId": "s",
            "uuid": "a1",
            "message": {
                "role": "assistant",
                "id": "msg_1",
                "model": "claude-opus-4-8",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "Write",
                        "input": {"file_path": "/tmp/note.txt", "content": "hello"},
                    }
                ],
            },
        }
        rejected = {
            "type": "user",
            "sessionId": "s",
            "uuid": "u1",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "is_error": True,
                        "content": "denied",
                    }
                ],
            },
        }
        return assistant, rejected

    def test_deny_emits_terminating_result(self):
        # RR1: after a deny the CLI writes no turn_duration; the transport must
        # synthesize a terminating result (subtype=success, is_error=False) so
        # receive_response() does not deadlock.
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # Simulate the watcher having answered "deny" for Write.
            t._pending_denied_tools = ["Write"]
            t._deny_terminated = True
            assistant, rejected = self._deny_transcript()
            await t._emit_line(json.dumps(assistant).encode())
            await t._emit_line(json.dumps(rejected).encode())
            return _drain(t)

        msgs = anyio.run(_test)
        results = [m for m in msgs if m["type"] == "result"]
        assert len(results) == 1
        r = results[0]
        assert r["subtype"] == "success"
        assert r["is_error"] is False

    def test_deny_records_baseline_denial_shape(self):
        # RR2: permission_denials carries {tool_name, tool_use_id, tool_input}
        # with the real id and full original input recovered from the transcript.
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._pending_denied_tools = ["Write"]
            t._deny_terminated = True
            assistant, rejected = self._deny_transcript()
            await t._emit_line(json.dumps(assistant).encode())
            await t._emit_line(json.dumps(rejected).encode())
            return _drain(t)

        msgs = anyio.run(_test)
        r = [m for m in msgs if m["type"] == "result"][0]
        assert r["permission_denials"] == [
            {
                "tool_name": "Write",
                "tool_use_id": "toolu_abc",
                "tool_input": {"file_path": "/tmp/note.txt", "content": "hello"},
            }
        ]

    def test_deny_excludes_rejected_tool_result_from_num_turns(self):
        # RR5: the rejected tool_result is not a real round-trip, so num_turns
        # is 1 (just the turn), not 2.
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._pending_denied_tools = ["Write"]
            t._deny_terminated = True
            assistant, rejected = self._deny_transcript()
            await t._emit_line(json.dumps(assistant).encode())
            await t._emit_line(json.dumps(rejected).encode())
            return _drain(t)

        msgs = anyio.run(_test)
        r = [m for m in msgs if m["type"] == "result"][0]
        assert r["num_turns"] == 1

    def test_deny_does_not_double_emit_on_late_turn_duration(self):
        # If a turn_duration somehow arrives after the deny result, no second
        # result is emitted.
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._pending_denied_tools = ["Write"]
            t._deny_terminated = True
            assistant, rejected = self._deny_transcript()
            await t._emit_line(json.dumps(assistant).encode())
            await t._emit_line(json.dumps(rejected).encode())
            await t._emit_line(
                json.dumps(
                    {"type": "system", "subtype": "turn_duration", "uuid": "td"}
                ).encode()
            )
            return _drain(t)

        msgs = anyio.run(_test)
        assert len([m for m in msgs if m["type"] == "result"]) == 1

    def test_non_deny_error_tool_result_still_counts(self):
        # A genuine tool error (not a permission deny) is a real round-trip and
        # is NOT recorded as a denial.
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # No pending deny -> the error tool_result is a normal round-trip.
            assistant, rejected = self._deny_transcript()
            await t._emit_line(json.dumps(assistant).encode())
            await t._emit_line(json.dumps(rejected).encode())
            await t._emit_line(
                json.dumps(
                    {"type": "system", "subtype": "turn_duration", "uuid": "td"}
                ).encode()
            )
            return _drain(t)

        msgs = anyio.run(_test)
        r = [m for m in msgs if m["type"] == "result"][0]
        assert r["permission_denials"] == []
        assert r["num_turns"] == 2  # the error tool_result counts as a round-trip


class TestPerTurnInit:
    """RR4: a fresh system/init message leads every turn."""

    def test_init_emitted_before_each_subsequent_turn(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # connect() already emitted the turn-1 init; subsequent turns get one
            # from _type_prompt. Simulate three prompt submissions.
            import unittest.mock as mock

            with (
                mock.patch.object(t, "_warmup", new=_noop),
                mock.patch.object(t, "_pty_write", new=_noop),
            ):
                for _ in range(3):
                    await t._type_prompt("hi")
            return _drain(t)

        msgs = anyio.run(_test)
        inits = [
            m for m in msgs if m.get("type") == "system" and m.get("subtype") == "init"
        ]
        # Turn 1's init is emitted at connect (not exercised here); turns 2 and 3
        # each emit one from _type_prompt.
        assert len(inits) == 2


async def _noop(*args, **kwargs):
    return None


# --------------------------------------------------------------------------- #
# Dedup / compaction tolerance
# --------------------------------------------------------------------------- #


class TestDedup:
    def test_duplicate_uuid_emitted_once(self):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            line = json.dumps(
                {
                    "type": "assistant",
                    "uuid": "dup",
                    "sessionId": "s",
                    "message": {
                        "role": "assistant",
                        "model": "m",
                        "content": [{"type": "text", "text": "x"}],
                    },
                }
            ).encode()
            await t._emit_line(line)
            await t._emit_line(line)  # re-read after a compaction reset
            return _drain(t)

        msgs = anyio.run(_test)
        assert sum(1 for m in msgs if m["type"] == "assistant") == 1

    def test_multiblock_same_id_blocks_all_survive(self, tmp_path):
        """RV1: the interactive transcript writes each content block of one
        assistant message as a SEPARATE record sharing one message.id (distinct
        uuids) -- [thinking], then [text], then [tool_use]. The baseline emits
        ONE AssistantMessage per block-record (per-block granularity, live A/B
        verified), so all three blocks must surface (the RV1 bug dropped every
        block after the first); usage is counted once across the shared id.
        """

        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"

            def rec(uid, block):
                return {
                    "type": "assistant",
                    "sessionId": "s",
                    "uuid": uid,
                    "message": {
                        "role": "assistant",
                        "id": "msg_X",
                        "model": "claude-opus-4-8",
                        "content": [block],
                        "stop_reason": "tool_use",
                        "usage": {"input_tokens": 100, "output_tokens": 5},
                    },
                }

            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        rec(
                            "u1",
                            {
                                "type": "thinking",
                                "thinking": "let me plan",
                                "signature": "sig",
                            },
                        ),
                        rec("u2", {"type": "text", "text": "I'll write files"}),
                        rec(
                            "u3",
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Write",
                                "input": {"file_path": "a.txt", "content": "1"},
                            },
                        ),
                        # A user tool_result ends the assistant message -> flush.
                        {
                            "type": "user",
                            "sessionId": "s",
                            "uuid": "u4",
                            "message": {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": "toolu_1",
                                        "content": "ok",
                                    }
                                ],
                            },
                        },
                        {"type": "system", "subtype": "turn_duration"},
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        # One AssistantMessage per block-record (baseline per-block granularity):
        # the thinking, text, and tool_use blocks each surface (the RV1 bug
        # surfaced only the first). Collect their block types in order.
        assistants = [m for m in msgs if m["type"] == "assistant"]
        block_types = [b["type"] for m in assistants for b in m["message"]["content"]]
        assert block_types == ["thinking", "text", "tool_use"]
        assert "tool_use" in block_types  # the headline RV1 regression
        # The tool_use block carries the real id + full input.
        tool_use = next(
            b
            for m in assistants
            for b in m["message"]["content"]
            if b["type"] == "tool_use"
        )
        assert tool_use["id"] == "toolu_1"
        assert tool_use["input"] == {"file_path": "a.txt", "content": "1"}
        # Ordering: the tool_use assistant record comes BEFORE the user
        # tool_result, matching the baseline stream.
        order = [m["type"] for m in msgs if m["type"] in ("assistant", "user")]
        assert order == ["assistant", "assistant", "assistant", "user"]
        # Usage counted once (single message.id), not x3.
        result = next(m for m in msgs if m["type"] == "result")
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 5


# --------------------------------------------------------------------------- #
# Multimodal content warning
# --------------------------------------------------------------------------- #


class TestMultimodalWarning:
    def test_non_text_blocks_warn_and_text_still_typed(self, caplog):
        async def _test():
            t = make_transport()
            t._warmed_up = True
            typed: list[str] = []

            async def fake_type(text: str) -> None:
                typed.append(text)

            t._type_prompt = fake_type  # type: ignore[method-assign]
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._ready = True
            with caplog.at_level("WARNING"):
                await t.write(
                    json.dumps(
                        {
                            "type": "user",
                            "message": {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "describe"},
                                    {"type": "image", "source": {}},
                                ],
                            },
                        }
                    )
                    + "\n"
                )
            return typed, [r.message for r in caplog.records]

        typed, warnings = anyio.run(_test)
        assert typed == ["describe"]
        assert any("non-text content" in w for w in warnings)


# --------------------------------------------------------------------------- #
# atexit cleanup registration
# --------------------------------------------------------------------------- #


class TestAtexitCleanup:
    def test_terminate_process_discards_from_active_set(self):
        t = make_transport()
        pty_cli._ACTIVE_CHILDREN.add(t)
        t._terminate_process()  # no process/fd; must be a safe no-op
        assert t not in pty_cli._ACTIVE_CHILDREN


# --------------------------------------------------------------------------- #
# R1: model_usage wire-key / parser round-trip
# --------------------------------------------------------------------------- #


class TestModelUsageParserRoundTrip:
    def test_modelusage_key_populates_resultmessage(self, tmp_path):
        """The synthesized result must populate ResultMessage.model_usage (R1).

        message_parser reads data.get("modelUsage") (camelCase); a snake_case
        model_usage key parses to None.
        """
        from claude_agent_sdk._internal.message_parser import parse_message
        from claude_agent_sdk.types import ResultMessage

        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "s",
                            "uuid": "a1",
                            "message": {
                                "role": "assistant",
                                "id": "msg_1",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": "done"}],
                                "stop_reason": "end_turn",
                                "usage": {
                                    "input_tokens": 1_000_000,
                                    "output_tokens": 0,
                                },
                            },
                        },
                        {
                            "type": "system",
                            "subtype": "turn_duration",
                            "durationMs": 100,
                        },
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._input_ended = True
            with anyio.fail_after(5):
                await t._tail_loop()
            return _drain(t)

        msgs = anyio.run(_test)
        result_data = next(m for m in msgs if m["type"] == "result")
        result_data.setdefault("session_id", "s")
        parsed = parse_message(result_data)
        assert isinstance(parsed, ResultMessage)
        # The whole point of R1: not None at the consumer.
        assert parsed.model_usage is not None
        assert parsed.model_usage["claude-opus-4-8"]["costUSD"] == 5.0


# --------------------------------------------------------------------------- #
# R4: interrupt synthesizes a terminating result
# --------------------------------------------------------------------------- #


class TestInterruptResult:
    def test_interrupt_emits_terminating_result(self):
        """interrupt() must synthesize a result so receive_response() ends (R4)."""

        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # Pretend a turn is in flight with some accumulated usage.
            t._result_emitted = False
            import time as _time

            t._turn_start_time = _time.monotonic()
            t._turn_usage.add("msg_1", "claude-opus-4-8", {"input_tokens": 1_000_000})
            t._turn_tool_results = 1
            with patch.object(t, "_pty_write", new=_noop_async):
                await t._handle_control_request(
                    {
                        "type": "control_request",
                        "request_id": "r",
                        "request": {"subtype": "interrupt"},
                    }
                )
            return _drain(t)

        msgs = anyio.run(_test)
        # ACK control_response is success.
        ctrl = next(m for m in msgs if m["type"] == "control_response")
        assert ctrl["response"]["subtype"] == "success"
        # A terminating result is synthesized, mirroring the stream-json baseline.
        result = next(m for m in msgs if m["type"] == "result")
        assert result["subtype"] == "error_during_execution"
        assert result["is_error"] is True
        assert result["result"] is None
        assert result["stop_reason"] is None
        assert result["num_turns"] == 2  # tool_results(1) + 1
        # Accumulated usage carried through.
        assert result["usage"]["input_tokens"] == 1_000_000

    def test_interrupt_does_not_double_emit(self):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._result_emitted = True  # a result already went out this turn
            with patch.object(t, "_pty_write", new=_noop_async):
                await t._handle_control_request(
                    {
                        "type": "control_request",
                        "request_id": "r",
                        "request": {"subtype": "interrupt"},
                    }
                )
            return _drain(t)

        msgs = anyio.run(_test)
        assert not any(m["type"] == "result" for m in msgs)

    def test_idle_interrupt_is_noop_and_does_not_poison_next_turn(self):
        """RW2: interrupt() with no turn in flight must NOT synthesize a result.

        Otherwise the spurious error_during_execution sits in the buffer and the
        NEXT real turn's receive_response() returns it first -- corrupting that
        turn. The baseline treats an idle interrupt as harmless.
        """

        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._result_emitted = False
            t._turn_start_time = None  # no active turn
            with patch.object(t, "_pty_write", new=_noop_async):
                await t._handle_control_request(
                    {
                        "type": "control_request",
                        "request_id": "r",
                        "request": {"subtype": "interrupt"},
                    }
                )
            return _drain(t)

        msgs = anyio.run(_test)
        # The ACK still goes out, but NO stale result is buffered.
        assert any(m["type"] == "control_response" for m in msgs)
        assert not any(m["type"] == "result" for m in msgs)


# --------------------------------------------------------------------------- #
# RW1: empty / whitespace-only prompt terminates instead of hanging
# --------------------------------------------------------------------------- #


class TestEmptyPromptResult:
    @pytest.mark.parametrize("prompt", ["", "   ", "\n\t  "])
    def test_empty_prompt_synthesizes_terminating_result(self, prompt):
        """RW1: an empty/whitespace prompt cannot be submitted to the TUI, so we
        synthesize a terminating success result (matching the baseline subtype)
        rather than letting receive_response() hang forever."""

        async def _test():
            from unittest.mock import AsyncMock

            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            typed = AsyncMock()
            with (
                patch.object(t, "_warmup", new=_noop_async),
                patch.object(t, "_pty_write", new=_noop_async),
                patch.object(t, "_type_prompt", new=typed),
            ):
                await t._handle_user_message(
                    {"message": {"role": "user", "content": prompt}}
                )
            # _type_prompt must NOT be called for an empty prompt.
            typed.assert_not_called()
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["subtype"] == "success"
        assert result["is_error"] is False
        assert result["permission_denials"] == []

    def test_nonempty_prompt_still_typed(self):
        async def _test():
            from unittest.mock import AsyncMock

            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            typed = AsyncMock()
            with patch.object(t, "_type_prompt", new=typed):
                await t._handle_user_message(
                    {"message": {"role": "user", "content": "hello"}}
                )
            typed.assert_called_once()
            return _drain(t)

        msgs = anyio.run(_test)
        # A real turn was submitted -> no synthetic result yet.
        assert not any(m["type"] == "result" for m in msgs)


# --------------------------------------------------------------------------- #
# RW3: resume/continue restores context (flag logic + tail offset)
# --------------------------------------------------------------------------- #


class TestResumeDropIn:
    def test_resume_does_not_auto_append_session_id(self):
        """RW3(1): with --resume and no caller session_id, the auto id must NOT
        be appended (it would override the resume target), matching baseline."""
        from claude_agent_sdk._internal.transport import _cli_command

        cmd = _cli_command.build_command(
            DEFAULT_CLI,
            ClaudeAgentOptions(resume="old-sid"),
            "auto-sid",
        )
        assert "--resume" in cmd
        assert cmd[cmd.index("--resume") + 1] == "old-sid"
        assert "--session-id" not in cmd

    def test_continue_does_not_auto_append_session_id(self):
        from claude_agent_sdk._internal.transport import _cli_command

        cmd = _cli_command.build_command(
            DEFAULT_CLI,
            ClaudeAgentOptions(continue_conversation=True),
            "auto-sid",
        )
        assert "--continue" in cmd
        assert "--session-id" not in cmd

    def test_explicit_session_id_wins_with_resume(self):
        from claude_agent_sdk._internal.transport import _cli_command

        cmd = _cli_command.build_command(
            DEFAULT_CLI,
            ClaudeAgentOptions(resume="old-sid", session_id="explicit"),
            "auto-sid",
        )
        assert cmd[cmd.index("--session-id") + 1] == "explicit"

    def test_new_session_still_auto_appends_session_id(self):
        from claude_agent_sdk._internal.transport import _cli_command

        cmd = _cli_command.build_command(
            DEFAULT_CLI,
            ClaudeAgentOptions(),
            "auto-sid",
        )
        assert cmd[cmd.index("--session-id") + 1] == "auto-sid"

    def test_resume_transcript_path_targets_resume_file(self):
        t = make_transport(resume="11111111-1111-1111-1111-111111111111")
        path = t._compute_transcript_path()
        assert path.name == "11111111-1111-1111-1111-111111111111.jsonl"

    def test_resume_tail_starts_past_preexisting_records(self, tmp_path):
        """RW3(2): when resuming an existing transcript, the tail loop must start
        at the file's current size so the prior turn's trailing turn_duration is
        NOT replayed as a stale result for the new turn."""

        async def _test():
            t = make_transport(resume="sid")
            path = tmp_path / "sid.jsonl"
            # A pre-existing prior turn, ending with its turn_duration.
            path.write_text(
                "\n".join(
                    json.dumps(line)
                    for line in [
                        {
                            "type": "assistant",
                            "sessionId": "sid",
                            "uuid": "old1",
                            "message": {
                                "role": "assistant",
                                "id": "msg_old",
                                "model": "claude-opus-4-8",
                                "content": [{"type": "text", "text": "DONE."}],
                            },
                        },
                        {
                            "type": "system",
                            "subtype": "turn_duration",
                            "sessionId": "sid",
                            "uuid": "olddur",
                            "durationMs": 1000,
                        },
                    ]
                )
                + "\n"
            )
            t._transcript_path = path
            # Simulate connect() seeding the offset past the existing records.
            t._initial_tail_offset = path.stat().st_size
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            t._closed = False
            t._input_ended = True  # let the loop stop once a result is emitted

            # Run the tail loop briefly; the new turn appends fresh records.
            async def _appender():
                with path.open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "type": "assistant",
                                "sessionId": "sid",
                                "uuid": "new1",
                                "message": {
                                    "role": "assistant",
                                    "id": "msg_new",
                                    "model": "claude-opus-4-8",
                                    "content": [{"type": "text", "text": "4271"}],
                                },
                            }
                        )
                        + "\n"
                    )
                    f.write(
                        json.dumps(
                            {
                                "type": "system",
                                "subtype": "turn_duration",
                                "sessionId": "sid",
                                "uuid": "newdur",
                                "durationMs": 500,
                            }
                        )
                        + "\n"
                    )

            async with anyio.create_task_group() as tg:
                tg.start_soon(t._tail_loop)
                await anyio.sleep(0.15)
                await _appender()
                with anyio.move_on_after(3):
                    while not t._result_emitted:
                        await anyio.sleep(0.05)
                t._closed = True
            return _drain(t)

        msgs = anyio.run(_test)
        # The old turn's DONE. text and old turn_duration must NOT be replayed.
        assert not any(
            m.get("type") == "assistant"
            and any(
                b.get("text") == "DONE."
                for b in m.get("message", {}).get("content", [])
            )
            for m in msgs
        )
        # Exactly one result, and it carries the NEW turn's text.
        results = [m for m in msgs if m.get("type") == "result"]
        assert len(results) == 1
        assert results[0]["result"] == "4271"


async def _noop_async(*_args, **_kwargs):
    return None


# --------------------------------------------------------------------------- #
# R8: --include-hook-events is honored at the CLI level
# --------------------------------------------------------------------------- #


class TestIncludeHookEvents:
    def test_flag_passed_to_cli(self):
        from claude_agent_sdk._internal.transport import _cli_command

        cmd = _cli_command.build_command(
            DEFAULT_CLI,
            ClaudeAgentOptions(include_hook_events=True),
            "sess",
        )
        assert "--include-hook-events" in cmd

    def test_flag_absent_by_default(self):
        from claude_agent_sdk._internal.transport import _cli_command

        cmd = _cli_command.build_command(DEFAULT_CLI, ClaudeAgentOptions(), "sess")
        assert "--include-hook-events" not in cmd


# --------------------------------------------------------------------------- #
# R9: bounded _seen_uuids
# --------------------------------------------------------------------------- #


class TestSeenUuidsBounded:
    def test_seen_uuids_capped(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"
            # Emit more distinct-uuid records than the cap; assistant records are
            # the simplest record type that goes through the dedup path.
            n = pty_cli._SEEN_UUIDS_MAX + 50
            lines = [
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": "s",
                        "uuid": f"u{i}",
                        "message": {
                            "role": "assistant",
                            "id": f"m{i}",
                            "model": "claude-opus-4-8",
                            "content": [{"type": "text", "text": "x"}],
                        },
                    }
                )
                for i in range(n)
            ]
            # Terminate the tail loop with a turn_duration so it stops spinning.
            lines.append(
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "turn_duration",
                        "uuid": "td",
                        "durationMs": 1,
                    }
                )
            )
            path.write_text("\n".join(lines) + "\n")
            t._transcript_path = path
            t._out_send, t._out_recv = anyio.create_memory_object_stream(n + 10)
            t._input_ended = True
            with anyio.fail_after(15):
                await t._tail_loop()
            return len(t._seen_uuids)

        size = anyio.run(_test)
        assert size <= pty_cli._SEEN_UUIDS_MAX


# --------------------------------------------------------------------------- #
# API-monitor traffic enrichment (C1 / R3 / R7 / RV2)
# --------------------------------------------------------------------------- #


class TestApiMonitorEnrichment:
    """The transport enriches the result from intercepted /v1/messages traffic.

    These drive the transport's monitor callback directly with synthetic call
    records (no sockets, no model) and assert the synthesized result reflects the
    captured traffic: cost/usage summed across ALL calls including the helper
    model (C1/R3), real summed api duration (R7), and non-2xx status (C1).
    """

    def _opus_call(self) -> dict:
        return {
            "path": "/v1/messages",
            "method": "POST",
            "status": 200,
            "duration_ms": 1200,
            "model": "claude-opus-4-8",
            "request": {"model": "claude-opus-4-8"},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            "stop_reason": "end_turn",
            "content_blocks": [{"type": "text", "text": "done"}],
            "partial_text": "done",
        }

    def _haiku_helper_call(self) -> dict:
        # The auxiliary title-generation call the transcript NEVER records (R3).
        return {
            "path": "/v1/messages",
            "method": "POST",
            "status": 200,
            "duration_ms": 300,
            "model": "claude-haiku-4-5",
            "request": {"model": "claude-haiku-4-5"},
            "usage": {"input_tokens": 1_000_000, "output_tokens": 0},
            "stop_reason": "end_turn",
            "content_blocks": [{"type": "text", "text": "A title"}],
            "partial_text": "A title",
        }

    def test_cost_and_model_usage_include_helper_call(self, tmp_path):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            # Simulate the monitor observing the opus turn + the haiku helper.
            t._on_api_call(self._opus_call())
            t._on_api_call(self._haiku_helper_call())
            await t._emit_result(
                {"type": "system", "subtype": "turn_duration", "durationMs": 9}
            )
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        # opus 1M input @ $5/Mtok = 5.0; haiku 1M input @ $1/Mtok = 1.0 -> 6.0.
        assert result["total_cost_usd"] == pytest.approx(6.0)
        # Both models appear in model_usage -- the helper line is no longer lost.
        assert set(result["modelUsage"].keys()) == {
            "claude-opus-4-8",
            "claude-haiku-4-5",
        }
        assert result["modelUsage"]["claude-haiku-4-5"]["costUSD"] == pytest.approx(1.0)
        # usage summed across BOTH calls.
        assert result["usage"]["input_tokens"] == 2_000_000

    def test_duration_api_ms_is_summed_real_timing(self, tmp_path):
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._on_api_call(self._opus_call())  # 1200ms
            t._on_api_call(self._haiku_helper_call())  # 300ms
            await t._emit_result(
                {"type": "system", "subtype": "turn_duration", "durationMs": 9}
            )
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        # Real summed per-call API time (R7), not the wall-clock durationMs (9).
        assert result["duration_api_ms"] == 1500
        assert result["duration_ms"] == 9

    def test_terminal_api_error_status_surfaced(self, tmp_path):
        # The turn ENDS on a non-2xx call -> surface it (C1).
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._on_api_call(self._opus_call())  # a 200
            err = self._opus_call()
            err["status"] = 529  # overloaded -- the LAST call fails
            err["usage"] = None
            t._on_api_call(err)
            await t._emit_result(
                {"type": "system", "subtype": "turn_duration", "durationMs": 9}
            )
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["api_error_status"] == 529

    def test_recovered_transient_error_not_surfaced(self, tmp_path):
        # A 529 the CLI then retried successfully (last call is 200) is NOT a
        # turn failure, so api_error_status is not set (faithful to baseline).
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            err = self._opus_call()
            err["status"] = 529
            err["usage"] = None
            t._on_api_call(err)
            t._on_api_call(self._opus_call())  # 200 retry succeeds
            await t._emit_result(
                {"type": "system", "subtype": "turn_duration", "durationMs": 9}
            )
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result.get("api_error_status") is None

    def test_falls_back_to_transcript_usage_without_traffic(self, tmp_path):
        # When the monitor observed nothing (no traffic), the result still uses
        # the transcript-derived accumulator -- no regression.
        async def _test():
            t = make_transport()
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._turn_usage.add(
                "msg_1",
                "claude-opus-4-8",
                {"input_tokens": 1_000_000, "output_tokens": 0},
            )
            await t._emit_result(
                {"type": "system", "subtype": "turn_duration", "durationMs": 9}
            )
            return _drain(t)

        msgs = anyio.run(_test)
        result = next(m for m in msgs if m["type"] == "result")
        assert result["total_cost_usd"] == pytest.approx(5.0)
        # No traffic -> duration_api_ms stays the wall-clock fallback.
        assert result["duration_api_ms"] == 9


class TestRecoverToolInputRV2:
    """RV2: can_use_tool receives the FULL tool input from intercepted traffic."""

    def test_full_input_recovered_by_tool_name(self):
        t = make_transport()
        # Simulate the monitor observing the assistant's tool_use in the response
        # while the permission dialog is still blocking.
        t._on_api_call(
            {
                "path": "/v1/messages",
                "method": "POST",
                "status": 200,
                "duration_ms": 5,
                "model": "claude-opus-4-8",
                "request": {"model": "claude-opus-4-8"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "tool_use",
                "content_blocks": [
                    {
                        "type": "tool_use",
                        "id": "toolu_42",
                        "name": "Write",
                        "input": {"file_path": "/tmp/x.txt", "content": "hi"},
                    }
                ],
                "partial_text": "",
            }
        )
        full_input, tool_use_id = t._recover_tool_input("Write")
        assert full_input == {"file_path": "/tmp/x.txt", "content": "hi"}
        assert tool_use_id == "toolu_42"

    def test_unknown_tool_returns_none(self):
        t = make_transport()
        assert t._recover_tool_input("Write") == (None, None)

    def test_decide_permission_passes_full_input_to_callback(self):
        from claude_agent_sdk import PermissionResultAllow
        from claude_agent_sdk._internal.transport.pty_question import (
            DetectedQuestion,
            QuestionOption,
        )

        seen: dict = {}

        async def cb(tool_name, tool_input, context):
            seen["name"] = tool_name
            seen["input"] = tool_input
            seen["tool_use_id"] = context.tool_use_id
            return PermissionResultAllow()

        async def _test():
            t = make_transport(can_use_tool=cb)
            t._on_api_call(
                {
                    "path": "/v1/messages",
                    "method": "POST",
                    "status": 200,
                    "duration_ms": 5,
                    "model": "claude-opus-4-8",
                    "request": {},
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                    "stop_reason": "tool_use",
                    "content_blocks": [
                        {
                            "type": "tool_use",
                            "id": "toolu_99",
                            "name": "Bash",
                            "input": {"command": "ls -la /etc"},
                        }
                    ],
                    "partial_text": "",
                }
            )
            question = DetectedQuestion(
                kind="permission",
                tool="Bash",
                target="ls",  # the TUI's scraped (partial) signal
                question="Allow Bash?",
                options=[QuestionOption(index=1, label="Yes")],
            )
            decision = await t._decide_permission(question)
            return decision

        decision = anyio.run(_test)
        assert decision == "allow"
        # The callback got the FULL input (RV2), not the scraped {target:'ls'}.
        assert seen["input"] == {"command": "ls -la /etc"}
        assert seen["tool_use_id"] == "toolu_99"
        assert seen["name"] == "Bash"


# --------------------------------------------------------------------------- #
# RL9 / RL10 / RL11: traffic-derived enrichments from the API monitor
# --------------------------------------------------------------------------- #


def _api_record(**kwargs: object) -> dict:
    """Build a minimal /v1/messages call record as the monitor tees it."""
    record: dict = {
        "path": "/v1/messages",
        "method": "POST",
        "status": 200,
        "duration_ms": 10,
        "model": None,
        "request": None,
        "usage": None,
        "stop_reason": None,
        "content_blocks": [],
        "partial_text": "",
        "sse_events": [],
    }
    record.update(kwargs)
    return record


class TestStreamEventsRL9:
    """RL9: stream_event messages reconstructed from SSE, gated on the option."""

    def test_emits_stream_events_when_option_set(self):
        async def _test():
            t = make_transport(include_partial_messages=True)
            t._out_send, t._out_recv = anyio.create_memory_object_stream(50)
            t._observed_session_id = "sess-1"
            events = [
                {"type": "message_start", "message": {"id": "msg_1"}},
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "PONG"},
                },
                {"type": "message_stop"},
            ]
            t._on_api_call(_api_record(sse_events=events))
            return [t._out_recv.receive_nowait() for _ in range(3)]

        msgs = anyio.run(_test)
        assert all(m["type"] == "stream_event" for m in msgs)
        # Same shape as the stream-json baseline StreamEvent.
        assert [m["event"]["type"] for m in msgs] == [
            "message_start",
            "content_block_delta",
            "message_stop",
        ]
        for m in msgs:
            assert m["session_id"] == "sess-1"
            assert m["parent_tool_use_id"] is None
            assert isinstance(m["uuid"], str) and m["uuid"]

    def test_emits_nothing_when_option_unset(self):
        async def _test():
            t = make_transport()  # include_partial_messages defaults to False
            t._out_send, t._out_recv = anyio.create_memory_object_stream(50)
            events = [{"type": "message_start", "message": {"id": "x"}}]
            t._on_api_call(_api_record(sse_events=events))
            with pytest.raises(anyio.WouldBlock):
                t._out_recv.receive_nowait()

        anyio.run(_test)

    def test_no_events_for_non_streaming_response(self):
        async def _test():
            t = make_transport(include_partial_messages=True)
            t._out_send, t._out_recv = anyio.create_memory_object_stream(50)
            t._on_api_call(_api_record(sse_events=[]))
            with pytest.raises(anyio.WouldBlock):
                t._out_recv.receive_nowait()

        anyio.run(_test)


class TestToolCatalogRL10:
    """RL10: init / get_server_info tools+model from the request catalog."""

    def test_init_uses_observed_tools_and_model(self):
        t = make_transport(allowed_tools=["OnlyOption"])
        t._on_api_call(
            _api_record(
                request={
                    "model": "claude-opus-4-8",
                    "tools": [
                        {"name": "Read"},
                        {"name": "Write"},
                        {"name": "Bash"},
                    ],
                }
            )
        )
        init = t._build_init_data()
        assert init["tools"] == ["Read", "Write", "Bash"]
        assert init["model"] == "claude-opus-4-8"

    def test_init_falls_back_to_options_before_first_request(self):
        t = make_transport(allowed_tools=["Read"], model="claude-sonnet-4-6")
        init = t._build_init_data()
        assert init["tools"] == ["Read"]
        assert init["model"] == "claude-sonnet-4-6"

    def test_server_info_carries_observed_tools_and_model(self):
        t = make_transport()
        t._on_api_call(
            _api_record(
                request={
                    "model": "claude-opus-4-8",
                    "tools": [{"name": "Read"}, {"name": "Edit"}],
                }
            )
        )
        info = t._build_server_info()
        assert info["tools"] == ["Read", "Edit"]
        assert info["model"] == "claude-opus-4-8"
        assert info["models"] == ["claude-opus-4-8"]

    def test_catalog_dedups_and_ignores_unnamed_tools(self):
        t = make_transport()
        t._on_api_call(
            _api_record(
                request={
                    "tools": [
                        {"name": "Read"},
                        {"name": "Read"},  # dup
                        {"type": "no_name"},  # ignored
                        {"name": "Write"},
                    ]
                }
            )
        )
        assert t._observed_tools == ["Read", "Write"]


class TestContextUsageRL11:
    """RL11: get_context_usage derived from the latest per-call token counts."""

    def test_payload_from_latest_usage(self):
        t = make_transport()
        t._on_api_call(
            _api_record(
                model="claude-opus-4-8",
                usage={
                    "input_tokens": 100,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 50,
                    "output_tokens": 5,
                },
            )
        )
        payload = t._context_usage_payload()
        # totalTokens is the input side (uncached + cache read + cache write).
        assert payload["totalTokens"] == 1150
        assert payload["model"] == "claude-opus-4-8"
        assert payload["rawMaxTokens"] == 1_000_000
        assert payload["maxTokens"] == 1_000_000
        assert payload["percentage"] == pytest.approx(0.115)
        # Unobservable breakdowns are empty, not fabricated.
        assert payload["categories"] == []
        assert payload["mcpTools"] == []

    def test_error_response_does_not_blank_context(self):
        t = make_transport()
        t._on_api_call(
            _api_record(model="claude-opus-4-8", usage={"input_tokens": 200})
        )
        # A later non-2xx call must not overwrite the live figure.
        t._on_api_call(_api_record(status=429, usage={"input_tokens": 1}))
        payload = t._context_usage_payload()
        assert payload["totalTokens"] == 200

    def test_control_request_returns_context_usage(self):
        async def _test():
            t = make_transport()
            t._ready = True
            t._out_send, t._out_recv = anyio.create_memory_object_stream(10)
            t._on_api_call(
                _api_record(model="claude-opus-4-8", usage={"input_tokens": 42})
            )
            await t.write(
                json.dumps(
                    {
                        "type": "control_request",
                        "request_id": "ctx-1",
                        "request": {"subtype": "get_context_usage"},
                    }
                )
                + "\n"
            )
            return t._out_recv.receive_nowait()

        msg = anyio.run(_test)
        assert msg["type"] == "control_response"
        assert msg["response"]["subtype"] == "success"
        assert msg["response"]["response"]["totalTokens"] == 42
