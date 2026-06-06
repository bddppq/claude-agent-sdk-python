"""PTY-based transport that drives the Claude Code CLI in interactive mode.

Unlike :class:`SubprocessCLITransport`, which speaks the bidirectional
``stream-json`` protocol over plain pipes (and historically ``--print``), this
transport launches the *interactive* CLI attached to a pseudo-terminal (PTY).
It then:

* types user prompts into the PTY (as a real terminal would), and
* reads the model's responses by **tailing the session transcript** the CLI
  writes to ``<config>/projects/<sanitized-cwd>/<session-id>.jsonl``.

The transcript records are translated back into the same message dictionaries
the rest of the SDK already understands (``assistant`` / ``user`` / ``result``
/ ``system``), so consumers and ``message_parser`` are unaffected.

POSIX only -- PTYs are not available on Windows.
"""

import contextlib
import fcntl
import json
import logging
import os
import pty
import signal
import struct
import termios
import time
import tty
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from pathlib import Path
from subprocess import Popen
from typing import Any

import anyio

from ..._errors import CLIConnectionError, CLINotFoundError
from ..._version import __version__
from ...types import ClaudeAgentOptions
from .._task_compat import TaskHandle, spawn_detached
from ..sessions import _canonicalize_path, _get_projects_dir, _sanitize_path
from . import Transport
from .subprocess_cli import SubprocessCLITransport

logger = logging.getLogger(__name__)

# Transcript record ``type`` values that carry no SDK-visible message. These are
# internal bookkeeping entries the CLI writes alongside the conversation.
_SKIP_TRANSCRIPT_TYPES = frozenset(
    {
        "queue-operation",
        "last-prompt",
        "ai-title",
        "attachment",
        "mode",
        "permission-mode",
        "file-history-snapshot",
        "summary",
    }
)

# Keystrokes sent into the PTY. A bare carriage return submits the current
# prompt in the TUI; ESC interrupts an in-progress turn.
_SUBMIT = b"\r"
_INTERRUPT = b"\x1b"


def _translate_transcript_entry(
    entry: dict[str, Any], session_id: str
) -> dict[str, Any] | None:
    """Translate a transcript ``.jsonl`` record into an SDK message dict.

    Returns ``None`` for records that should not surface to consumers (internal
    bookkeeping, or the plain-text user prompt we typed ourselves).
    """
    entry_type = entry.get("type")
    sid = entry.get("sessionId") or session_id

    if entry_type == "assistant":
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        return {
            "type": "assistant",
            "message": _sanitize_assistant_message(message),
            "session_id": sid,
            "uuid": entry.get("uuid"),
            "parent_tool_use_id": None,
        }

    if entry_type == "user":
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        # Skip the echo of the prompt we typed; only surface user records that
        # carry tool results (the structured part of a tool round-trip).
        if isinstance(content, str):
            return None
        if isinstance(content, list) and not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            return None
        return {
            "type": "user",
            "message": message,
            "session_id": sid,
            "uuid": entry.get("uuid"),
            "parent_tool_use_id": None,
        }

    if entry_type == "system" and entry.get("subtype") == "turn_duration":
        # The CLI writes a ``turn_duration`` system record when an assistant
        # turn finishes. The interactive transcript has no ``result`` record,
        # so synthesize one -- it is the signal the rest of the SDK uses to
        # know a turn completed.
        duration = entry.get("durationMs", 0)
        return {
            "type": "result",
            "subtype": "success",
            "duration_ms": duration,
            "duration_api_ms": duration,
            "is_error": False,
            "num_turns": entry.get("messageCount", 1),
            "session_id": sid,
            "result": None,
            "uuid": entry.get("uuid"),
        }

    return None


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Backfill fields the parser requires that the transcript may omit.

    ``thinking`` blocks must carry a ``signature``; the transcript sometimes
    omits it. Patch a default so ``message_parser`` does not raise.
    """
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                block.setdefault("signature", "")
    return message


class PtyCLITransport(Transport):
    """Transport that runs the interactive CLI over a PTY and tails its transcript."""

    def __init__(
        self,
        prompt: str | AsyncIterable[dict[str, Any]],
        options: ClaudeAgentOptions,
    ):
        if os.name != "posix":
            raise CLIConnectionError(
                "PtyCLITransport requires a POSIX platform (PTYs are unavailable "
                "on Windows)."
            )
        self._prompt = prompt
        self._options = options
        self._cli_path: str | None = (
            str(options.cli_path) if options.cli_path is not None else None
        )
        self._cwd = str(options.cwd) if options.cwd else str(Path.cwd())
        self._session_id = options.session_id or str(uuid.uuid4())

        self._proc: Popen[bytes] | None = None
        self._master_fd: int | None = None
        self._transcript_path: Path | None = None

        self._out_send: Any = None
        self._out_recv: Any = None
        self._drain_task: TaskHandle | None = None
        self._tail_task: TaskHandle | None = None

        self._ready = False
        self._closed = False
        self._input_ended = False
        self._result_emitted = False
        self._warmed_up = False
        self._spawn_time = 0.0
        # Tail of recent PTY output, kept for diagnostics if the CLI exits
        # before producing any transcript output.
        self._recent_output = b""

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        if self._proc is not None:
            return

        if self._cli_path is None:
            self._cli_path = await anyio.to_thread.run_sync(self._find_cli)

        # Interactive mode shows first-run onboarding (theme/login) screens that
        # block programmatic input. Mark onboarding complete so the CLI drops
        # straight into the prompt. Best-effort and non-destructive.
        await anyio.to_thread.run_sync(self._ensure_onboarding_complete)

        cmd = self._build_command()

        master_fd, slave_fd = pty.openpty()
        # Give the TUI a sane window and put the line discipline in raw mode so
        # our writes are not echoed back into the transcript-tailing path.
        with contextlib.suppress(OSError):
            fcntl.ioctl(
                slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0)
            )
        with contextlib.suppress(OSError):
            tty.setraw(slave_fd)

        try:
            self._proc = Popen(  # noqa: S603 - cmd is built from vetted options
                cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self._cwd,
                env=self._build_env(),
                close_fds=True,
                preexec_fn=os.setsid,  # own process group so we can signal it
            )
        except FileNotFoundError as e:
            os.close(master_fd)
            os.close(slave_fd)
            if not Path(self._cwd).exists():
                raise CLIConnectionError(
                    f"Working directory does not exist: {self._cwd}"
                ) from e
            raise CLINotFoundError(f"Claude Code not found at: {self._cli_path}") from e
        except Exception as e:
            os.close(master_fd)
            os.close(slave_fd)
            raise CLIConnectionError(f"Failed to start Claude Code: {e}") from e

        os.close(slave_fd)  # parent keeps only the master end
        self._master_fd = master_fd
        self._spawn_time = time.monotonic()

        self._transcript_path = self._compute_transcript_path()

        self._out_send, self._out_recv = anyio.create_memory_object_stream[
            dict[str, Any]
        ](max_buffer_size=1000)

        self._drain_task = spawn_detached(self._drain_loop())
        self._tail_task = spawn_detached(self._tail_loop())

        self._ready = True

        # Surface a minimal init message so consumers can read the session id
        # before the first turn lands in the transcript.
        await self._out_send.send(
            {
                "type": "system",
                "subtype": "init",
                "session_id": self._session_id,
                "cwd": self._cwd,
                "uuid": str(uuid.uuid4()),
            }
        )

    def _find_cli(self) -> str:
        # Reuse the well-tested discovery logic from the pipe transport.
        return SubprocessCLITransport(prompt="", options=self._options)._find_cli()

    def _build_env(self) -> dict[str, str]:
        # Filter CLAUDECODE so the child does not believe it is nested inside a
        # parent Claude Code (see subprocess_cli for rationale).
        inherited = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        env = {
            **inherited,
            "CLAUDE_CODE_ENTRYPOINT": "sdk-py-pty",
            **self._options.env,
            "CLAUDE_AGENT_SDK_VERSION": __version__,
        }
        env["PWD"] = self._cwd
        # Running as root, the CLI refuses bypassPermissions /
        # --dangerously-skip-permissions unless IS_SANDBOX marks a contained
        # environment. The CLI only accepts the exact value "1", so normalize
        # any inherited value (e.g. "yes") unless the caller set one explicitly.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            env["IS_SANDBOX"] = self._options.env.get("IS_SANDBOX", "1")
        return env

    def _ensure_onboarding_complete(self) -> None:
        """Clear interactive gates that would block programmatic input.

        The interactive CLI blocks the prompt behind two screens that a real
        user dismisses by hand:

        * first-run onboarding (theme/login) -- gated by
          ``hasCompletedOnboarding``;
        * a per-folder trust dialog -- gated by
          ``projects[<cwd>].hasTrustDialogAccepted``.

        We merge both flags into the CLI config, preserving all existing data.
        Pre-seeding the config is more robust than timing blind keystrokes
        against a full-screen TUI.
        """
        config_dir = self._options.env.get("CLAUDE_CONFIG_DIR") or os.environ.get(
            "CLAUDE_CONFIG_DIR"
        )
        config_path = (
            Path(config_dir) / ".claude.json"
            if config_dir
            else Path.home() / ".claude.json"
        )
        try:
            data: dict[str, Any] = {}
            if config_path.exists():
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    data = json.loads(config_path.read_text(encoding="utf-8"))

            changed = False
            if data.get("hasCompletedOnboarding") is not True:
                data["hasCompletedOnboarding"] = True
                changed = True

            projects = data.setdefault("projects", {})
            # The CLI keys projects by both the literal cwd and its realpath in
            # different code paths; trust both so the dialog never appears.
            for key in {self._cwd, os.path.realpath(self._cwd)}:
                project = projects.setdefault(key, {})
                if project.get("hasTrustDialogAccepted") is not True:
                    project["hasTrustDialogAccepted"] = True
                    project.setdefault("projectOnboardingSeenCount", 1)
                    changed = True

            if not changed:
                return
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(data), encoding="utf-8")
            logger.debug("Pre-seeded onboarding/trust flags in %s", config_path)
        except OSError:
            logger.debug("Could not update CLI config flags", exc_info=True)

    def _build_command(self) -> list[str]:
        """Build the interactive CLI command.

        Reuses :meth:`SubprocessCLITransport._build_command` for full option
        coverage, then strips the ``stream-json`` I/O flags that only apply to
        the headless protocol so the CLI starts its interactive TUI instead.
        """
        if self._cli_path is None:
            raise CLINotFoundError("CLI path not resolved. Call connect() first.")

        # Ensure a known session id so we can locate the transcript file.
        opts = self._options
        if not opts.session_id:
            opts = replace(opts, session_id=self._session_id)
        helper = SubprocessCLITransport(prompt="", options=opts)
        helper._cli_path = self._cli_path
        raw = helper._build_command()

        drop_with_value = {"--output-format", "--input-format"}
        drop_flag = {
            "--verbose",
            "--include-partial-messages",
            "--include-hook-events",
            "--session-mirror",
        }
        cmd: list[str] = []
        skip_next = False
        for tok in raw:
            if skip_next:
                skip_next = False
                continue
            if tok in drop_with_value:
                skip_next = True
                continue
            if tok in drop_flag:
                continue
            cmd.append(tok)
        return cmd

    def _compute_transcript_path(self) -> Path:
        project_dir = _get_projects_dir(
            env_override=self._options.env
        ) / _sanitize_path(_canonicalize_path(self._cwd))
        return project_dir / f"{self._session_id}.jsonl"

    # ------------------------------------------------------------------ #
    # Background loops
    # ------------------------------------------------------------------ #

    async def _drain_loop(self) -> None:
        """Continuously read and discard PTY output.

        The TUI keeps rendering to the terminal; if we never drain the master
        end the pty buffer fills and the CLI blocks. We do not parse this
        output -- the transcript file is the source of truth.
        """
        assert self._master_fd is not None
        fd = self._master_fd
        while not self._closed:
            try:
                data = await anyio.to_thread.run_sync(
                    self._blocking_read, fd, abandon_on_cancel=True
                )
            except anyio.get_cancelled_exc_class():
                raise
            except Exception:
                break
            if not data:
                break
            # Keep a bounded tail for diagnostics (e.g. early-exit error text).
            self._recent_output = (self._recent_output + data)[-4096:]

    @staticmethod
    def _blocking_read(fd: int) -> bytes:
        try:
            return os.read(fd, 65536)
        except OSError:
            return b""

    async def _tail_loop(self) -> None:
        """Tail the transcript file and translate new records into messages."""
        assert self._transcript_path is not None
        path = self._transcript_path
        offset = 0
        buffer = b""
        try:
            while not self._closed:
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0

                if size > offset:
                    with path.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    for raw_line in lines:
                        await self._emit_line(raw_line)

                # One-shot termination: input has been closed and the turn
                # finished, so there is nothing more to wait for.
                if self._input_ended and self._result_emitted:
                    break

                # Process exited and the transcript is fully read.
                if (
                    self._proc is not None
                    and self._proc.poll() is not None
                    and size <= offset
                ):
                    await self._handle_early_exit(self._proc.returncode)
                    break

                await anyio.sleep(0.1)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("Transcript tail loop failed", exc_info=True)
        finally:
            if self._out_send is not None:
                with contextlib.suppress(Exception):
                    self._out_send.close()

    async def _handle_early_exit(self, returncode: int | None) -> None:
        """If the CLI exited non-zero without producing a turn, report it.

        Surfaces a result message with ``is_error`` so consumers see a real
        failure (e.g. an unusable flag) instead of an empty, silent stream.
        """
        if returncode in (None, 0) or self._result_emitted:
            return
        import re

        text = re.sub(
            r"\x1b\[[0-9;?]*[A-Za-z]",
            "",
            self._recent_output.decode("utf-8", "replace"),
        )
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())[
            -500:
        ]
        error_result = {
            "type": "result",
            "subtype": "error_during_execution",
            "duration_ms": 0,
            "duration_api_ms": 0,
            "is_error": True,
            "num_turns": 0,
            "session_id": self._session_id,
            "result": text or f"Claude Code exited with code {returncode}",
            "errors": [text] if text else [f"exit code {returncode}"],
            "uuid": str(uuid.uuid4()),
        }
        self._result_emitted = True
        if self._out_send is not None:
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError
            ):
                await self._out_send.send(error_result)

    async def _emit_line(self, raw_line: bytes) -> None:
        line = raw_line.strip()
        if not line:
            return
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(entry, dict):
            return
        if entry.get("type") in _SKIP_TRANSCRIPT_TYPES:
            return
        message = _translate_transcript_entry(entry, self._session_id)
        if message is None:
            return
        if message.get("type") == "result":
            self._result_emitted = True
        if self._out_send is not None:
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError
            ):
                await self._out_send.send(message)

    # ------------------------------------------------------------------ #
    # Transport interface
    # ------------------------------------------------------------------ #

    async def write(self, data: str) -> None:
        if not self._ready:
            raise CLIConnectionError("PtyCLITransport is not ready for writing")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            msg_type = obj.get("type")
            if msg_type == "control_request":
                await self._handle_control_request(obj)
            elif msg_type == "user":
                await self._handle_user_message(obj)
            # Other frame types have no interactive equivalent; ignore them.

    async def _handle_control_request(self, obj: dict[str, Any]) -> None:
        request = obj.get("request", {})
        subtype = request.get("subtype")
        request_id = obj.get("request_id")

        if subtype == "interrupt":
            await self._pty_write(_INTERRUPT)

        # The interactive CLI does not speak the SDK control protocol, so we
        # acknowledge control requests locally to keep the handshake flowing.
        response = {
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": request_id,
                "response": {},
            },
        }
        if self._out_send is not None:
            with contextlib.suppress(
                anyio.BrokenResourceError, anyio.ClosedResourceError
            ):
                await self._out_send.send(response)

    async def _handle_user_message(self, obj: dict[str, Any]) -> None:
        message = obj.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            text = ""
        if text.strip():
            await self._type_prompt(text)

    async def _type_prompt(self, text: str) -> None:
        await self._warmup()
        # Type the prompt as plain keystrokes. Newlines are collapsed to spaces
        # because a bare CR submits the prompt in the TUI; sending the body and
        # the submit key as separate writes lets the editor settle in between.
        body = " ".join(text.splitlines()).encode("utf-8")
        await self._pty_write(body)
        await anyio.sleep(0.3)
        await self._pty_write(_SUBMIT)

    async def _warmup(self) -> None:
        """Wait for the TUI to render and dismiss the startup notification.

        The interactive CLI shows a transient startup toast that swallows the
        first Enter. Letting the UI settle and sending one dismissal Enter
        (an empty prompt is a no-op) ensures the subsequent submit registers.
        """
        if self._warmed_up:
            return
        self._warmed_up = True
        elapsed = time.monotonic() - self._spawn_time
        if elapsed < 3.0:
            await anyio.sleep(3.0 - elapsed)
        await self._pty_write(_SUBMIT)  # dismiss startup toast
        await anyio.sleep(0.5)

    async def _pty_write(self, data: bytes) -> None:
        if self._master_fd is None:
            return
        fd = self._master_fd

        def _write_all() -> None:
            view = data
            while view:
                written = os.write(fd, view)
                view = view[written:]

        with contextlib.suppress(OSError):
            await anyio.to_thread.run_sync(_write_all)

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._read_messages_impl()

    async def _read_messages_impl(self) -> AsyncIterator[dict[str, Any]]:
        if self._out_recv is None:
            raise CLIConnectionError("Not connected")
        async for message in self._out_recv:
            yield message

    async def end_input(self) -> None:
        # Interactive mode has no stdin EOF semantics; record that the caller is
        # done so the tail loop can terminate after the current turn (one-shot).
        self._input_ended = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready = False

        for task in (self._tail_task, self._drain_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    await task.wait()
        self._tail_task = None
        self._drain_task = None

        if self._out_send is not None:
            with contextlib.suppress(Exception):
                self._out_send.close()

        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            with contextlib.suppress(Exception):
                await anyio.to_thread.run_sync(self._wait_proc, abandon_on_cancel=True)

        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None

        self._proc = None

    def _wait_proc(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=5)

    def is_ready(self) -> bool:
        return self._ready
