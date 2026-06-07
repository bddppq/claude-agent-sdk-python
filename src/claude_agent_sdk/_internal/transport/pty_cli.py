"""PTY-based transport that drives the Claude Code CLI in interactive mode.

This is the SDK's transport. Rather than speaking the headless ``stream-json``
protocol over pipes (the former ``--print`` mode), it launches the
*interactive* CLI attached to a pseudo-terminal (PTY). It then:

* types user prompts into the PTY (as a real terminal would), and
* reads the model's responses by **tailing the session transcript** the CLI
  writes to ``<config>/projects/<sanitized-cwd>/<session-id>.jsonl``.

The transcript records are translated back into the same message dictionaries
the rest of the SDK already understands (``assistant`` / ``user`` / ``result``
/ ``system``), so consumers and ``message_parser`` are unaffected.

POSIX only -- PTYs are not available on Windows.

Unsupported options (interactive mode has no equivalent SDK channel). These are
rejected up front by :meth:`PtyCLITransport._validate_options` with an
actionable error rather than failing silently or hanging:

* ``can_use_tool`` -- the interactive CLI never delivers tool-permission
  requests to the SDK (that round-trip exists only in stream-json), so the
  callback can never be invoked. Tools still run; they just can't be gated via
  this callback. Use ``permission_mode`` / ``allowed_tools`` / settings instead;
* ``hooks`` -- programmatic hook callbacks require the bidirectional protocol;
* in-process ``mcp_servers`` of ``type="sdk"`` -- reachable only over the
  control protocol (external stdio/http/sse MCP servers still work);
* ``session_store`` -- relied on ``transcript_mirror`` stdout frames;
* ``permission_prompt_tool_name``.

``include_partial_messages`` and ``include_hook_events`` are accepted but warned
about (no partial/hook-event records exist in the transcript).
"""

import atexit
import contextlib
import fcntl
import json
import logging
import os
import pty
import re
import signal
import struct
import tempfile
import termios
import time
import tty
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from subprocess import Popen
from typing import Any

import anyio

from ..._errors import CLIConnectionError, CLINotFoundError
from ...types import ClaudeAgentOptions
from .._task_compat import TaskHandle, spawn_detached
from ..sessions import _canonicalize_path, _get_projects_dir, _sanitize_path
from . import Transport, _cli_command
from ._usage import TurnUsageAccumulator
from .pty_question import (
    SCREEN_COLS,
    SCREEN_ROWS,
    DetectedQuestion,
    parse_question,
)

logger = logging.getLogger(__name__)

# Track live child process groups + master fds so we can terminate them when the
# parent Python process exits, mirroring the old subprocess transport's
# parent-exit cleanup. Prevents orphaned interactive ``claude`` processes when a
# caller crashes or exits before awaiting close().
_ACTIVE_CHILDREN: "set[PtyCLITransport]" = set()


def _kill_active_children() -> None:
    for transport in list(_ACTIVE_CHILDREN):
        with contextlib.suppress(Exception):
            transport._terminate_process()
    _ACTIVE_CHILDREN.clear()


atexit.register(_kill_active_children)

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
# prompt in the TUI; ESC interrupts an in-progress turn (app:interrupt);
# shift+tab (CSI Z, "back-tab") cycles the permission mode (chat:cycleMode).
_SUBMIT = b"\r"
_INTERRUPT = b"\x1b"
_SHIFT_TAB = b"\x1b[Z"

# Bracketed-paste guards. Wrapping the prompt in these makes the TUI insert the
# text verbatim into the editor -- preserving newlines and NOT interpreting a
# leading "/", "!", or "#" as a slash/bash/memory command -- after which a
# single carriage return submits it. This is critical for prompt fidelity.
_PASTE_START = b"\x1b[200~"
_PASTE_END = b"\x1b[201~"

# Seconds to let the interactive TUI render before the first prompt is typed.
# The CLI shows a transient startup toast that swallows the first Enter; we wait
# this long, then send one dismissal Enter. Module-level so tests can shrink it.
_WARMUP_SECONDS = 3.0

# Order the TUI cycles through on shift+tab. bypassPermissions is not part of
# the cycle (it is only reachable via launch flag), so it cannot be set live.
_PERMISSION_CYCLE = ("default", "acceptEdits", "plan")


def _translate_transcript_entry(
    entry: dict[str, Any], session_id: str
) -> dict[str, Any] | None:
    """Translate a transcript ``.jsonl`` record into an SDK message dict.

    Returns ``None`` for records that should not surface to consumers (internal
    bookkeeping, or the plain-text user prompt we typed ourselves).
    """
    entry_type = entry.get("type")
    sid = entry.get("sessionId") or session_id

    # parent_tool_use_id carries sub-agent attribution; the transcript records
    # it under a few possible keys.
    parent = entry.get("parent_tool_use_id") or entry.get("parentToolUseId")

    if entry_type == "assistant":
        message = entry.get("message")
        if not isinstance(message, dict):
            return None
        return {
            "type": "assistant",
            "message": _sanitize_assistant_message(message),
            "session_id": sid,
            "uuid": entry.get("uuid"),
            "parent_tool_use_id": parent,
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
            "parent_tool_use_id": parent,
        }

    return None


def _sanitize_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    """Coerce a transcript assistant message into a parser-safe shape.

    The transcript is an internal format with weaker guarantees than the old
    ``--print`` stream, so defend against records the parser would reject:

    * ``content`` as a string -> wrap in a single text block;
    * ``content`` missing/non-list -> empty list;
    * ``thinking`` blocks missing ``signature`` -> default it;
    * ``model`` missing -> default to ``"unknown"`` (parser requires it).
    """
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = [{"type": "text", "text": content}]
    elif not isinstance(content, list):
        message["content"] = []
    else:
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                block.setdefault("signature", "")
    message.setdefault("model", "unknown")
    return message


def _extract_text(message: dict[str, Any]) -> str:
    """Concatenate the text blocks of an assistant message."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


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
        # Track the live permission mode so set_permission_mode can compute how
        # many shift+tab cycles are needed to reach a target.
        self._permission_mode: str = options.permission_mode or "default"

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

        # Optional terminal-screen emulator (pyte) for introspecting blocking
        # questions rendered in the TUI. Stays None when pyte is unavailable, in
        # which case detect_question() simply returns None.
        self._question_screen: Any = None
        self._question_stream: Any = None

        # Serializes keystroke sequences so concurrent prompts / control actions
        # don't interleave bytes into the PTY.
        self._write_lock = anyio.Lock()
        # Per-turn state used to synthesize a faithful ``result`` message.
        self._turn_count = 0
        self._turn_text = ""
        self._turn_usage = TurnUsageAccumulator()
        self._turn_is_error = False
        self._turn_error_text: str | None = None
        self._turn_subtype: str | None = None
        self._turn_stop_reason: str | None = None
        self._turn_model: str | None = None
        # Count of assistant + tool-result messages in the current turn, used as
        # a fallback for num_turns when the CLI's messageCount is unavailable.
        self._turn_messages = 0
        # Permission denials observed in the current turn (answered "deny" via
        # the TUI question detector). Surfaced on the result like stream-json.
        self._turn_permission_denials: list[dict[str, Any]] = []
        # Real session id discovered from transcript records (resume/fork can
        # make the CLI use an id different from the one we generated).
        self._observed_session_id: str | None = None
        # Cached server-info payload for get_server_info() / initialize ack.
        self._server_info: dict[str, Any] | None = None
        # Dedup transcript records by uuid so a mid-session compaction/rewrite
        # (which resets the read offset) cannot re-emit already-seen messages.
        self._seen_uuids: set[str] = set()

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> None:
        if self._proc is not None:
            return

        self._validate_options()

        if self._cli_path is None:
            self._cli_path = await anyio.to_thread.run_sync(_cli_command.find_cli)

        if not os.environ.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            await _cli_command.check_claude_version(self._cli_path)

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
                slave_fd,
                termios.TIOCSWINSZ,
                struct.pack("HHHH", SCREEN_ROWS, SCREEN_COLS, 0, 0),
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
        _ACTIVE_CHILDREN.add(self)

        self._transcript_path = self._compute_transcript_path()

        self._out_send, self._out_recv = anyio.create_memory_object_stream[
            dict[str, Any]
        ](max_buffer_size=1000)

        self._init_question_screen()

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

    def _validate_options(self) -> None:
        """Reject options the interactive transport cannot honor.

        Fails loudly up front instead of silently no-op-ing or hanging mid-turn.
        """
        o = self._options
        unsupported: list[str] = []
        if o.can_use_tool is not None:
            unsupported.append(
                "can_use_tool (the interactive CLI never sends tool-permission "
                "requests back to the SDK -- that round-trip exists only in the "
                "stream-json control protocol -- so the callback would never "
                "fire; gate tools with permission_mode / allowed_tools / settings)"
            )
        if o.hooks:
            unsupported.append(
                "hooks (programmatic hook callbacks require the bidirectional "
                "control protocol; use command hooks in settings instead)"
            )
        if o.permission_prompt_tool_name:
            unsupported.append("permission_prompt_tool_name")
        if o.session_store is not None:
            unsupported.append(
                "session_store (transcript mirroring relied on stream-json frames)"
            )
        if isinstance(o.mcp_servers, dict):
            sdk_servers = [
                name
                for name, cfg in o.mcp_servers.items()
                if isinstance(cfg, dict) and cfg.get("type") == "sdk"
            ]
            if sdk_servers:
                unsupported.append(
                    "in-process SDK MCP servers "
                    f"({', '.join(sdk_servers)}) -- external stdio/http/sse MCP "
                    "servers are still supported"
                )
        if unsupported:
            raise CLIConnectionError(
                "These options are not supported by the interactive transport:\n  - "
                + "\n  - ".join(unsupported)
            )

        # Accepted-but-inert observability flags: warn rather than fail.
        if o.include_partial_messages:
            logger.warning(
                "include_partial_messages has no effect with the interactive "
                "transport; no partial/stream_event records exist in the transcript."
            )
        if o.include_hook_events:
            logger.warning(
                "include_hook_events has no effect with the interactive transport."
            )
        if o.stderr is not None:
            logger.warning(
                "The stderr callback is not invoked by the interactive transport "
                "(the CLI's stderr is multiplexed onto the PTY)."
            )
        if o.max_buffer_size is not None:
            logger.debug(
                "max_buffer_size is ignored by the interactive transport "
                "(messages are read from the transcript file, not a pipe)."
            )

    def _build_env(self) -> dict[str, str]:
        return _cli_command.build_env(self._options, self._cwd, entrypoint="sdk-py-pty")

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
            # Write atomically (temp file + os.replace) so a concurrent CLI or
            # another SDK client racing this write cannot read or leave behind a
            # half-written / clobbered ~/.claude.json.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(config_path.parent), prefix=".claude.json.", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                Path(tmp_name).replace(config_path)
            except OSError:
                with contextlib.suppress(OSError):
                    Path(tmp_name).unlink()
                raise
            logger.debug("Pre-seeded onboarding/trust flags in %s", config_path)
        except OSError:
            logger.debug("Could not update CLI config flags", exc_info=True)

    def _build_command(self) -> list[str]:
        """Build the interactive CLI command from the configured options."""
        if self._cli_path is None:
            raise CLINotFoundError("CLI path not resolved. Call connect() first.")
        return _cli_command.build_command(
            self._cli_path, self._options, self._session_id
        )

    def _compute_transcript_path(self) -> Path:
        project_dir = _get_projects_dir(
            env_override=self._options.env
        ) / _sanitize_path(_canonicalize_path(self._cwd))
        return project_dir / f"{self._session_id}.jsonl"

    def _resolve_transcript_path(self) -> Path | None:
        """Locate the transcript the CLI is actually writing, or ``None``.

        Resolution is deliberately conservative -- returning ``None`` (keep
        waiting) is always safer than guessing another session's transcript:

        1. the exact computed path (the normal case);
        2. any ``<our-session-id>.jsonl`` anywhere under the projects dir
           (handles a long-cwd directory-hash mismatch; still keyed to *our*
           session id, so it can't match a different conversation);
        3. only when the CLI may have chosen a different id
           (``--fork-session`` / ``--resume`` / ``--continue``): the newest
           ``*.jsonl`` in *our own project dir* modified at/after spawn.

        Step 3 is scoped to our project directory and gated on those flags so a
        concurrent unrelated ``claude`` session in another directory is never
        picked up.
        """
        if self._transcript_path and self._transcript_path.exists():
            return self._transcript_path

        projects = _get_projects_dir(env_override=self._options.env)
        fname = f"{self._session_id}.jsonl"
        with contextlib.suppress(OSError):
            for p in projects.rglob(fname):
                return p

        o = self._options
        may_fork = bool(o.fork_session or o.resume or o.continue_conversation)
        project_dir = self._transcript_path.parent if self._transcript_path else None
        if not may_fork or project_dir is None or not project_dir.is_dir():
            return None

        best: Path | None = None
        best_mtime = self._spawn_time  # only files touched at/after spawn qualify
        with contextlib.suppress(OSError):
            for p in project_dir.glob("*.jsonl"):
                mtime = p.stat().st_mtime
                if mtime >= best_mtime:
                    best, best_mtime = p, mtime
        return best

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
            # Feed the terminal emulator so detect_question() can read the
            # current screen. Defensive suppress: a malformed escape must never
            # take down the drain loop.
            if self._question_stream is not None:
                with contextlib.suppress(Exception):
                    self._question_stream.feed(data)

    def _init_question_screen(self) -> None:
        """Create the pyte screen used by detect_question(), if pyte is present.

        Geometry matches the PTY winsize so the emulated screen mirrors what the
        CLI draws. Absence of pyte is non-fatal: detect_question() returns None.
        """
        try:
            import pyte
        except ImportError:
            self._question_screen = None
            self._question_stream = None
            return
        self._question_screen = pyte.Screen(SCREEN_COLS, SCREEN_ROWS)
        self._question_stream = pyte.ByteStream(self._question_screen)

    def detect_question(self) -> DetectedQuestion | None:
        """Return the question currently blocking the TUI, or None.

        Reconstructs the rendered screen and parses any tool-permission prompt,
        plan-approval, AskUserQuestion form, or app-level confirmation into a
        structured :class:`DetectedQuestion`. Requires the optional ``pyte``
        dependency (the ``pty-introspect`` extra); returns None without it.
        """
        screen = self._question_screen
        if screen is None:
            return None
        return parse_question([line.rstrip() for line in screen.display])

    @staticmethod
    def _blocking_read(fd: int) -> bytes:
        try:
            return os.read(fd, 65536)
        except OSError:
            return b""

    async def _tail_loop(self) -> None:
        """Tail the transcript file and translate new records into messages."""
        path: Path | None = None
        offset = 0
        buffer = b""
        try:
            while not self._closed:
                if path is None:
                    path = self._resolve_transcript_path()
                    if path is not None:
                        self._transcript_path = path

                size = 0
                if path is not None:
                    try:
                        size = path.stat().st_size
                    except OSError:
                        size = 0

                    # The transcript was truncated/compacted (rewritten shorter):
                    # re-read from the start. The uuid dedup in _emit_line keeps
                    # already-seen records from being emitted twice.
                    if size < offset:
                        offset = 0
                        buffer = b""

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

                # Process exited; drain any remaining content, then stop.
                if self._proc is not None and self._proc.poll() is not None:
                    if path is not None and size > offset:
                        await anyio.sleep(0.05)
                        continue
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
        text = re.sub(
            r"\x1b\[[0-9;?]*[A-Za-z]",
            "",
            self._recent_output.decode("utf-8", "replace"),
        )
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())[
            -500:
        ]
        self._result_emitted = True
        await self._send(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "duration_ms": 0,
                "duration_api_ms": 0,
                "is_error": True,
                "num_turns": self._turn_count,
                "session_id": self._session_id,
                "result": text or f"Claude Code exited with code {returncode}",
                "errors": [text] if text else [f"exit code {returncode}"],
                "uuid": str(uuid.uuid4()),
            }
        )

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
        entry_type = entry.get("type")
        if entry_type in _SKIP_TRANSCRIPT_TYPES:
            return

        # Track the real session id the CLI is using (resume/fork may differ
        # from our generated one). Used to keep session_id consistent on every
        # emitted message and on the result.
        sid = entry.get("sessionId")
        if isinstance(sid, str) and sid:
            self._observed_session_id = sid

        # Dedup by uuid so a compaction-triggered re-read can't double-emit.
        uid = entry.get("uuid")
        if isinstance(uid, str):
            if uid in self._seen_uuids:
                return
            self._seen_uuids.add(uid)

        # turn_duration is the turn-complete signal -> synthesize a result that
        # carries the turn's final text, accumulated usage, and error state.
        if entry_type == "system" and entry.get("subtype") == "turn_duration":
            await self._emit_result(entry)
            return

        message = _translate_transcript_entry(entry, self._session_id)
        if message is None:
            return

        if entry_type == "assistant":
            msg = message["message"]
            text = _extract_text(msg)
            if text:
                self._turn_text = text
            model = msg.get("model")
            if isinstance(model, str) and model and model != "unknown":
                self._turn_model = model
            stop_reason = msg.get("stop_reason")
            if isinstance(stop_reason, str):
                self._turn_stop_reason = stop_reason
            self._turn_usage.add(msg.get("id"), model, msg.get("usage"))
            # Each assistant message is a CLI "turn" (API round-trip); count
            # them so num_turns matches the stream-json baseline, which counts
            # API turns rather than user prompts.
            self._turn_messages += 1
            err = msg.get("error")
            if err or stop_reason == "refusal":
                self._turn_is_error = True
                self._turn_subtype = self._error_subtype(err, stop_reason)
        elif entry_type == "user":
            # Tool-result user records are also turns in the CLI's accounting.
            self._turn_messages += 1

        await self._send(message)

    @staticmethod
    def _error_subtype(error: Any, stop_reason: str | None) -> str:
        """Map an assistant error/stop_reason to a result subtype.

        Mirrors the stream-json result subtypes so consumers branching on
        ``subtype`` (refusal, max-turns, budget) keep working.
        """
        if stop_reason == "refusal":
            return "error_during_execution"
        if isinstance(error, str):
            low = error.lower()
            if "max_turns" in low or "max turns" in low:
                return "error_max_turns"
            if "budget" in low:
                return "error_max_budget_usd"
        return "error_during_execution"

    async def _emit_result(self, entry: dict[str, Any]) -> None:
        """Synthesize and emit a result from a ``turn_duration`` record."""
        self._turn_count += 1
        duration = entry.get("durationMs", 0)
        # num_turns: prefer the CLI's own messageCount for the turn (matches the
        # stream-json baseline, which counts API turns, not user prompts);
        # fall back to the assistant/tool-result messages we observed.
        message_count = entry.get("messageCount")
        if isinstance(message_count, int) and message_count > 0:
            num_turns = message_count
        else:
            num_turns = max(self._turn_messages, 1)

        subtype = (
            (self._turn_subtype or "error_during_execution")
            if self._turn_is_error
            else "success"
        )
        result: dict[str, Any] = {
            "type": "result",
            "subtype": subtype,
            "duration_ms": duration,
            # No per-API duration is recorded in the transcript; the turn's
            # wall-clock duration is the closest faithful value.
            "duration_api_ms": duration,
            "is_error": self._turn_is_error,
            "num_turns": num_turns,
            "session_id": self._observed_session_id
            or entry.get("sessionId")
            or self._session_id,
            "result": self._turn_text or None,
            "stop_reason": self._turn_stop_reason,
            "uuid": entry.get("uuid"),
        }
        if self._turn_usage.has_data():
            result["usage"] = self._turn_usage.aggregate_usage()
            cost = self._turn_usage.total_cost()
            if cost is not None:
                result["total_cost_usd"] = cost
            model_usage = self._turn_usage.model_usage()
            if model_usage is not None:
                result["model_usage"] = model_usage
        # permission_denials: empty list (faithful default; stream-json always
        # sent a list, never None, so formatting that iterates it works).
        result["permission_denials"] = list(self._turn_permission_denials)
        self._result_emitted = True
        await self._send(result)
        # Reset per-turn accumulators for the next turn.
        self._turn_text = ""
        self._turn_usage = TurnUsageAccumulator()
        self._turn_is_error = False
        self._turn_subtype = None
        self._turn_stop_reason = None
        self._turn_messages = 0
        self._turn_permission_denials = []

    async def _send(self, message: dict[str, Any]) -> None:
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

    # Control subtypes with a faithful interactive equivalent. Anything else is
    # answered with an error control_response (rather than a fake success) so
    # callers get a clear failure instead of a silent no-op.
    _SUPPORTED_CONTROLS = frozenset(
        {"initialize", "interrupt", "set_permission_mode", "set_model"}
    )

    async def _handle_control_request(self, obj: dict[str, Any]) -> None:
        """Map an SDK control request to its interactive-TUI equivalent.

        Supported: ``initialize`` (local ack), ``interrupt`` (ESC),
        ``set_permission_mode`` (shift+tab cycling), ``set_model`` (``/model``).
        Everything else (mcp_status, get_context_usage, mcp_reconnect,
        mcp_toggle, stop_task, rewind_files, ...) has no interactive channel
        that returns data, so it gets an explicit error control_response.
        """
        request = obj.get("request", {})
        subtype = request.get("subtype")
        request_id = obj.get("request_id")
        error: str | None = None

        try:
            if subtype == "interrupt":
                async with self._write_lock:
                    await self._pty_write(_INTERRUPT)
            elif subtype == "set_permission_mode":
                error = await self._set_permission_mode(request.get("mode"))
            elif subtype == "set_model":
                error = await self._set_model(request.get("model"))
            elif subtype not in self._SUPPORTED_CONTROLS:
                error = (
                    f"control request '{subtype}' is not supported by the "
                    "interactive transport"
                )
            # initialize needs no interactive action.
        except Exception as e:  # noqa: BLE001
            logger.debug("Interactive control action failed", exc_info=True)
            error = f"interactive control action failed: {e}"

        if error is not None:
            await self._send(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "error",
                        "request_id": request_id,
                        "error": error,
                    },
                }
            )
        else:
            await self._send(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": request_id,
                        "response": {},
                    },
                }
            )

    async def _set_permission_mode(self, mode: str | None) -> str | None:
        """Cycle the TUI permission mode to ``mode`` via shift+tab.

        Returns an error string if the mode can't be reached by cycling.
        """
        if mode is None or mode == self._permission_mode:
            return None
        if (
            mode not in _PERMISSION_CYCLE
            or self._permission_mode not in _PERMISSION_CYCLE
        ):
            # bypassPermissions / dontAsk / auto are not in the shift+tab cycle;
            # they can only be set via the launch flag, not live.
            return (
                f"permission mode {mode!r} cannot be set live over the "
                f"interactive transport (only {', '.join(_PERMISSION_CYCLE)} "
                "are reachable; set others via options.permission_mode)"
            )
        await self._warmup()
        async with self._write_lock:
            current = _PERMISSION_CYCLE.index(self._permission_mode)
            target = _PERMISSION_CYCLE.index(mode)
            steps = (target - current) % len(_PERMISSION_CYCLE)
            for _ in range(steps):
                await self._pty_write(_SHIFT_TAB)
                await anyio.sleep(0.1)
        self._permission_mode = mode
        return None

    async def _set_model(self, model: str | None) -> str | None:
        """Switch the model via the ``/model`` slash command."""
        if not model:
            return None
        await self._run_slash_command(f"/model {model}")
        return None

    async def _run_slash_command(self, command: str) -> None:
        await self._warmup()
        async with self._write_lock:
            await self._pty_write(command.encode("utf-8"))
            await anyio.sleep(0.2)
            await self._pty_write(_SUBMIT)

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
            # Warn (once) about structured blocks that can't be typed into a TUI.
            dropped = {
                b.get("type")
                for b in content
                if isinstance(b, dict) and b.get("type") not in ("text",)
            }
            if dropped:
                logger.warning(
                    "Dropping non-text content blocks the interactive transport "
                    "cannot send: %s",
                    ", ".join(sorted(str(d) for d in dropped)),
                )
        else:
            text = ""
        if text.strip():
            await self._type_prompt(text)

    async def _type_prompt(self, text: str) -> None:
        await self._warmup()
        # Bracketed paste makes the TUI insert the text verbatim, preserving
        # newlines and most special characters, after which a single CR submits.
        # Exception: a leading "/", "!" or "#" still triggers the TUI's
        # slash/bash/memory mode even when pasted (and Enter then fails to
        # submit), so prepend a single space in that one case. The model treats
        # leading whitespace as insignificant.
        if text[:1] in ("/", "!", "#"):
            text = " " + text
        payload = _PASTE_START + text.encode("utf-8") + _PASTE_END
        async with self._write_lock:
            await self._pty_write(payload)
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
        if elapsed < _WARMUP_SECONDS:
            await anyio.sleep(_WARMUP_SECONDS - elapsed)
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
        _ACTIVE_CHILDREN.discard(self)

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

    def _terminate_process(self) -> None:
        """Synchronously kill the child process group and close the master fd.

        Used by the ``atexit`` cleanup so a caller that crashes without awaiting
        ``close()`` does not leak the interactive ``claude`` process or its fd.
        """
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        if self._master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._master_fd)
            self._master_fd = None
        _ACTIVE_CHILDREN.discard(self)

    def is_ready(self) -> bool:
        return self._ready
