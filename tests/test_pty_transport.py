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

    def test_user_text_block_list_without_tool_result_is_skipped(self):
        entry = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            },
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
        assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py-pty"
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
        ["mcp_status", "get_context_usage", "rewind_files", "stop_task", "mcp_toggle"],
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
            ({"can_use_tool": lambda *a: None}, "can_use_tool"),
            (
                {"hooks": {"PreToolUse": [{"hooks": [lambda *a: None]}]}},
                "hooks",
            ),
            ({"permission_prompt_tool_name": "stdio"}, "permission_prompt_tool_name"),
        ],
    )
    def test_unsupported_options_raise(self, kwargs, needle):
        from claude_agent_sdk._errors import CLIConnectionError

        t = make_transport(**kwargs)
        with pytest.raises(CLIConnectionError) as exc:
            t._validate_options()
        assert needle in str(exc.value)

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

    def test_observability_flags_warn_not_raise(self, caplog):
        t = make_transport(include_partial_messages=True, include_hook_events=True)
        with caplog.at_level("WARNING"):
            t._validate_options()  # must not raise
        assert any("include_partial_messages" in r.message for r in caplog.records)


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
        assert result["num_turns"] == 1
        # usage summed across both assistant messages
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

    def test_multi_turn_increments_num_turns(self, tmp_path):
        async def _test():
            t = make_transport()
            path = tmp_path / "s.jsonl"

            def turn(n):
                return [
                    {
                        "type": "assistant",
                        "sessionId": "s",
                        "uuid": f"a{n}",
                        "message": {
                            "role": "assistant",
                            "model": "m",
                            "content": [{"type": "text", "text": f"r{n}"}],
                        },
                    },
                    {
                        "type": "system",
                        "subtype": "turn_duration",
                        "uuid": f"s{n}",
                    },
                ]

            lines = turn(1) + turn(2)
            path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
            t._transcript_path = path
            # not one-shot; stop by closing after reading
            t._out_send, t._out_recv = anyio.create_memory_object_stream(100)
            # Drive a single pass of the loop body via _emit_line directly.
            for x in lines:
                await t._emit_line(json.dumps(x).encode())
            return _drain(t)

        msgs = anyio.run(_test)
        results = [m for m in msgs if m["type"] == "result"]
        assert [r["num_turns"] for r in results] == [1, 2]


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
