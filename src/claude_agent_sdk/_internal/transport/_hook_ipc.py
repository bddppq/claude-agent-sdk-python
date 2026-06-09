"""Settings-hook IPC bridge for the interactive (PTY) transport.

This is the SDK-process half of the "win #3" hook channel. It lets the
interactive CLI -- which only knows how to run **command** hooks declared in
``settings.json`` -- drive the SDK's *programmatic* ``options.hooks`` callbacks
and route tool permissions through ``options.can_use_tool`` **deterministically**
(not by scraping the TUI dialog).

How the pieces fit together
---------------------------

1. :class:`HookIpcServer` listens on a localhost endpoint (a Unix domain socket
   when available, else a TCP loopback socket). It is started in the transport's
   ``connect()`` and stopped in ``close()``.
2. :func:`build_hooks_settings` synthesizes a ``settings.json`` ``hooks`` block
   that wires every relevant hook event to the shim command
   (``python -m claude_agent_sdk._internal.transport._hook_shim``). The shim's
   endpoint + auth token are injected via the child env (``CLAUDE_AGENT_SDK_HOOK_IPC``).
3. When the CLI fires a hook, it runs the shim, which forwards the hook-event
   JSON to this server. :meth:`HookIpcServer._handle` dispatches to the user's
   ``options.hooks`` callbacks (matching the old control-protocol hook shapes)
   and, for ``PreToolUse`` when ``can_use_tool`` is set, calls ``can_use_tool``
   and translates the result into the CLI's ``permissionDecision`` /
   ``updatedInput`` short-circuit.

Everything here is **fallback-safe**: if the server cannot start, the transport
simply does not wire the hook settings and keeps its existing TUI-watcher
behavior. If a dispatch raises, the server returns an empty ``{}`` (the CLI's
"no opinion"), so a turn never hangs or crashes on a hook failure.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import SocketStream

from ...types import (
    HookContext,
    HookMatcher,
    PermissionResultAllow,
    PermissionResultDeny,
)
from .._task_compat import TaskHandle, spawn_detached

logger = logging.getLogger(__name__)

ENV_HOOK_IPC = "CLAUDE_AGENT_SDK_HOOK_IPC"

# Hook events that carry a per-tool ``matcher`` (tool-name pattern). For these,
# the synthesized settings entry preserves the user's matcher; all other events
# are wired with a catch-all matcher.
_TOOL_MATCHED_EVENTS = frozenset(
    {"PreToolUse", "PostToolUse", "PostToolUseFailure", "PermissionRequest"}
)

# Cap a single inbound request so a misbehaving peer cannot exhaust memory.
_MAX_REQUEST_BYTES = 16 * 1024 * 1024


# Type of the can_use_tool callback as the transport passes it in. Kept loose so
# this module does not depend on the exact dataclass import beyond the result
# types it translates.
CanUseToolFn = Callable[..., Awaitable[Any]]


def _convert_hook_output_for_cli(hook_output: dict[str, Any]) -> dict[str, Any]:
    """Convert Python-safe field names to the CLI-expected names.

    Mirrors the old control-protocol transport: the Python SDK uses ``async_``
    and ``continue_`` to avoid keyword conflicts; the CLI expects ``async`` and
    ``continue`` in the hook-output JSON.
    """
    converted: dict[str, Any] = {}
    for key, value in hook_output.items():
        if key == "async_":
            converted["async"] = value
        elif key == "continue_":
            converted["continue"] = value
        else:
            converted[key] = value
    return converted


def shim_command() -> list[str]:
    """The argv that the CLI should run for each wired hook event."""
    return [sys.executable, "-m", "claude_agent_sdk._internal.transport._hook_shim"]


class HookIpcServer:
    """Localhost IPC endpoint that dispatches CLI hook events to SDK callbacks.

    Parameters
    ----------
    hooks:
        The user's ``options.hooks`` (``{event: [HookMatcher, ...]}``) or None.
    can_use_tool:
        The user's ``options.can_use_tool`` or None. When set, a synthetic
        ``PreToolUse`` hook routes tool permissions through it deterministically.
    permission_mode:
        Used only to annotate the synthetic ``PreToolUse`` hook input's
        ``permission_mode`` field if the CLI did not supply one.
    """

    def __init__(
        self,
        hooks: dict[str, list[HookMatcher]] | None,
        can_use_tool: CanUseToolFn | None,
        permission_mode: str | None = None,
        on_permission_decision: Callable[[str, str, str | None], None] | None = None,
    ) -> None:
        self._hooks = hooks or {}
        self._can_use_tool = can_use_tool
        self._permission_mode = permission_mode
        # Optional sink invoked with (tool_name, decision, tool_use_id) right
        # after a can_use_tool PreToolUse decision is made, so the transport can
        # record a hook-channel deny for result.permission_denials correlation
        # (the watcher path is not involved when the hook short-circuits the
        # dialog). Best-effort; runs on the IPC serve task (same event loop).
        self._on_permission_decision = on_permission_decision

        self._token = secrets.token_hex(16)
        self._listener: Any = None
        self._serve_handle: TaskHandle | None = None
        # Set once started: the spec string the shim parses (see _hook_shim).
        self._spec: str | None = None
        self._unix_path: str | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    @property
    def spec(self) -> str:
        if self._spec is None:
            raise RuntimeError("HookIpcServer.start() has not completed")
        return self._spec

    def wired_events(self) -> set[str]:
        """Events that this server will actually receive (for settings synthesis)."""
        events = set(self._hooks.keys())
        if self._can_use_tool is not None:
            events.add("PreToolUse")
        return events

    async def start(self) -> None:
        """Bind the IPC listener and begin serving. Raises on bind failure.

        Prefers a Unix domain socket (no exposed port, filesystem-permission
        scoped); falls back to a TCP loopback listener where AF_UNIX is
        unavailable (e.g. Windows).
        """
        if hasattr(anyio, "create_unix_listener") and hasattr(os, "AF_UNIX"):
            # Place the socket in a private temp dir with a random name so other
            # local users cannot connect to it (dir is 0700 by default).
            tmpdir = tempfile.mkdtemp(prefix="claude-agent-sdk-hook-")
            path = str(Path(tmpdir) / f"{secrets.token_hex(8)}.sock")
            try:
                self._listener = await anyio.create_unix_listener(path)
                self._unix_path = path
                self._spec = f"unix:{path}:{self._token}"
            except Exception:
                with suppress(OSError):
                    Path(tmpdir).rmdir()
                # Fall through to TCP below.
                self._listener = None

        if self._listener is None:
            from anyio.abc import SocketAttribute

            listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
            port = None
            for sub in getattr(listener, "listeners", [listener]):
                port = sub.extra(SocketAttribute.local_address)[1]
                break
            if port is None:
                with suppress(Exception):
                    await listener.aclose()
                raise RuntimeError("could not determine hook IPC listener port")
            self._listener = listener
            self._spec = f"tcp:127.0.0.1:{port}:{self._token}"

        self._serve_handle = spawn_detached(self._serve())

    async def _serve(self) -> None:
        assert self._listener is not None
        try:
            await self._listener.serve(self._handle_client)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("hook IPC serve loop ended", exc_info=True)

    async def stop(self) -> None:
        """Stop serving and release the listener + socket file (safe from any task)."""
        if self._serve_handle is not None:
            self._serve_handle.cancel()
            with suppress(Exception):
                await self._serve_handle.wait()
            self._serve_handle = None
        if self._listener is not None:
            with suppress(Exception):
                await self._listener.aclose()
            self._listener = None
        if self._unix_path is not None:
            sock_path = Path(self._unix_path)
            with suppress(OSError):
                sock_path.unlink()
            with suppress(OSError):
                sock_path.parent.rmdir()
            self._unix_path = None

    # ------------------------------------------------------------------ #
    # Per-connection handling
    # ------------------------------------------------------------------ #

    async def _handle_client(self, client: SocketStream) -> None:
        reply = b"{}\n"
        try:
            async with client:
                raw = await self._read_request(client)
                reply = await self._build_reply(raw)
                with suppress(Exception):
                    await client.send(reply)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("hook IPC connection failed", exc_info=True)

    @staticmethod
    async def _read_request(client: SocketStream) -> bytes:
        chunks: list[bytes] = []
        total = 0
        with suppress(anyio.EndOfStream):
            while True:
                chunk = await client.receive(65536)
                chunks.append(chunk)
                total += len(chunk)
                if b"\n" in chunk or total > _MAX_REQUEST_BYTES:
                    break
        return b"".join(chunks)

    async def _build_reply(self, raw: bytes) -> bytes:
        """Parse the request, dispatch, and return the newline-terminated reply.

        Always returns valid JSON + ``\\n``; on any failure returns ``{}`` so the
        CLI falls back to its normal (e.g. TUI dialog) behavior.
        """
        try:
            text = raw.decode("utf-8", "replace").strip()
            request = json.loads(text) if text else {}
        except Exception:
            return b"{}\n"

        if not isinstance(request, dict) or request.get("addr") != self._token:
            # Missing/incorrect auth token: refuse to dispatch.
            return b"{}\n"

        event = request.get("event")
        if not isinstance(event, dict):
            return b"{}\n"

        try:
            output = await self._dispatch(event)
        except Exception:
            logger.debug("hook dispatch raised; returning no-op", exc_info=True)
            return b"{}\n"

        try:
            return (json.dumps(output) + "\n").encode("utf-8")
        except Exception:
            return b"{}\n"

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    async def _dispatch(self, event: dict[str, Any]) -> dict[str, Any]:
        """Run user hook callbacks (+ can_use_tool for PreToolUse) for one event.

        Returns the merged hook-output JSON (already field-name-converted for the
        CLI). An empty dict means "no opinion".
        """
        event_name = event.get("hook_event_name")
        tool_name = event.get("tool_name")
        tool_use_id = event.get("tool_use_id")

        merged: dict[str, Any] = {}

        # 1) User-configured programmatic hooks for this event.
        for callback in self._matching_callbacks(event_name, tool_name):
            try:
                result = await callback(event, tool_use_id, HookContext(signal=None))
            except Exception:
                logger.debug("user hook callback raised; skipping", exc_info=True)
                continue
            if isinstance(result, dict) and result:
                merged.update(_convert_hook_output_for_cli(result))

        # 2) can_use_tool routing for PreToolUse: translate the SDK permission
        #    result into the CLI's permissionDecision / updatedInput short-circuit.
        if (
            event_name == "PreToolUse"
            and self._can_use_tool is not None
            and isinstance(tool_name, str)
        ):
            perm = await self._run_can_use_tool(event, tool_name, tool_use_id)
            if perm is not None:
                # A can_use_tool decision is authoritative for the permission
                # channel: it overrides any permission opinion a user hook set,
                # but we keep non-permission fields the user hook contributed.
                merged.setdefault("hookSpecificOutput", {})
                if not isinstance(merged.get("hookSpecificOutput"), dict):
                    merged["hookSpecificOutput"] = {}
                merged["hookSpecificOutput"].update(perm["hookSpecificOutput"])

        return merged

    def _matching_callbacks(
        self, event_name: Any, tool_name: Any
    ) -> list[Callable[..., Awaitable[Any]]]:
        matchers = self._hooks.get(event_name) if isinstance(event_name, str) else None
        if not matchers:
            return []
        callbacks: list[Callable[..., Awaitable[Any]]] = []
        for matcher in matchers:
            if self._matcher_matches(matcher.matcher, tool_name):
                callbacks.extend(matcher.hooks)
        return callbacks

    @staticmethod
    def _matcher_matches(pattern: str | None, tool_name: Any) -> bool:
        """Mirror the CLI's matcher semantics (utils/hooks.ts:matchesPattern).

        ``None``/``""``/``"*"`` match everything; a pipe-separated list matches
        any exact name; otherwise treat as a regex (fall back to exact on a bad
        pattern). Non-tool events have ``tool_name`` undefined -> only the
        catch-all matches.
        """
        if not pattern or pattern == "*":
            return True
        if not isinstance(tool_name, str):
            return False
        if "|" in pattern and all(c.isalnum() or c in "_|" for c in pattern):
            return tool_name in [p.strip() for p in pattern.split("|")]
        import re

        try:
            return re.search(pattern, tool_name) is not None
        except re.error:
            return tool_name == pattern

    def _notify_decision(self, tool_name: str, decision: str, tool_use_id: Any) -> None:
        """Tell the transport about a permission decision (best-effort)."""
        if self._on_permission_decision is None:
            return
        with suppress(Exception):
            self._on_permission_decision(
                tool_name,
                decision,
                tool_use_id if isinstance(tool_use_id, str) else None,
            )

    async def _run_can_use_tool(
        self,
        event: dict[str, Any],
        tool_name: str,
        tool_use_id: Any,
    ) -> dict[str, Any] | None:
        """Call can_use_tool and build the PreToolUse permission output.

        Returns a dict ``{"hookSpecificOutput": {...}}`` carrying the CLI's
        ``permissionDecision`` (+ ``updatedInput`` / ``permissionDecisionReason``),
        or ``None`` if the callback could not be run (so other channels apply).
        """
        # Import here to avoid a hard module-level dependency cycle and to keep
        # the public surface of this module small.
        from ...types import ToolPermissionContext

        tool_input = event.get("tool_input")
        if not isinstance(tool_input, dict):
            tool_input = {}

        context = ToolPermissionContext(
            tool_use_id=tool_use_id if isinstance(tool_use_id, str) else None,
        )

        assert self._can_use_tool is not None
        try:
            result = await self._can_use_tool(tool_name, tool_input, context)
        except Exception:
            logger.debug("can_use_tool raised in hook channel; denying", exc_info=True)
            self._notify_decision(tool_name, "deny", tool_use_id)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "can_use_tool callback raised",
                }
            }

        if isinstance(result, PermissionResultAllow):
            hook_specific: dict[str, Any] = {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
            if result.updated_input is not None:
                hook_specific["updatedInput"] = result.updated_input
            self._notify_decision(tool_name, "allow", tool_use_id)
            return {"hookSpecificOutput": hook_specific}

        if isinstance(result, PermissionResultDeny):
            self._notify_decision(tool_name, "deny", tool_use_id)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": result.message or "Denied by callback",
                }
            }

        # Defensive: a malformed return -> deny (matches the watcher's behavior of
        # treating non-Allow as deny).
        self._notify_decision(tool_name, "deny", tool_use_id)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "can_use_tool returned an invalid result",
            }
        }


def build_hooks_settings(
    server: HookIpcServer,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Synthesize the ``settings.json`` ``hooks`` block wiring events to the shim.

    Every event the server will handle (the user's configured hook events, plus
    ``PreToolUse`` when ``can_use_tool`` is set) gets a matcher entry whose single
    command hook runs the shim. Tool-matched events preserve the user's matcher
    string; the synthetic ``PreToolUse`` entry uses a catch-all matcher so it sees
    every tool. The result is suitable for merging into the ``--settings`` JSON.
    """
    shim = " ".join(_shell_quote(part) for part in shim_command())
    timeout = int(timeout_seconds) if timeout_seconds else None

    def _command_hook() -> dict[str, Any]:
        entry: dict[str, Any] = {"type": "command", "command": shim}
        if timeout is not None and timeout > 0:
            entry["timeout"] = timeout
        return entry

    hooks_block: dict[str, list[dict[str, Any]]] = {}

    # Preserve the user's matchers for events they configured (so per-tool
    # matchers still filter on the CLI side, exactly as command hooks would).
    user_hooks = server._hooks  # noqa: SLF001 - same package, intentional
    for event, matchers in user_hooks.items():
        entries: list[dict[str, Any]] = []
        for matcher in matchers:
            cfg: dict[str, Any] = {"hooks": [_command_hook()]}
            if matcher.matcher is not None and event in _TOOL_MATCHED_EVENTS:
                cfg["matcher"] = matcher.matcher
            entries.append(cfg)
        if entries:
            hooks_block[event] = entries

    # Synthetic PreToolUse hook for can_use_tool routing. If the user already
    # configured PreToolUse hooks above, the shim is already wired for that event
    # (one shim call dispatches BOTH the user hooks AND can_use_tool), so only add
    # a catch-all entry when there is none yet.
    if server._can_use_tool is not None and "PreToolUse" not in hooks_block:  # noqa: SLF001
        hooks_block["PreToolUse"] = [{"hooks": [_command_hook()]}]

    return hooks_block


def _shell_quote(part: str) -> str:
    """Quote a single argv element for the CLI's shell-string hook command.

    The CLI runs command hooks via ``spawn(cmd, [], {shell: true})`` -- i.e. the
    whole ``command`` string is parsed by ``/bin/sh`` (or Git Bash on Windows).
    So the synthesized command must be a properly shell-quoted string.
    """
    import shlex

    return shlex.quote(part)
