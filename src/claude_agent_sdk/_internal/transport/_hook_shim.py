"""Tiny, dependency-free hook shim launched by the interactive CLI.

The PTY transport wires the CLI's ``settings.json`` ``hooks`` entries to run
``python -m claude_agent_sdk._internal.transport._hook_shim``. The CLI executes
this command for each hook event (PreToolUse, PostToolUse, ...), feeding the
hook-event JSON on **stdin** and reading the hook-output JSON from **stdout**
(see the CLI's ``execCommandHook`` / ``parseHookOutput``).

This shim forwards the event to the SDK process's hook IPC server (a localhost
endpoint whose address the transport injects via the
``CLAUDE_AGENT_SDK_HOOK_IPC`` environment variable), waits for the SDK's decision,
and writes that decision back to stdout. The SDK process is where the user's
``options.hooks`` callbacks and ``options.can_use_tool`` actually run.

Design constraints (deliberately strict):

* **stdlib only.** This runs as a fresh ``python -m`` process spawned by the CLI;
  importing the full SDK (anyio, the transport stack, ...) would be slow and
  fragile. It uses only ``json``/``os``/``socket``/``struct``/``sys``.
* **Robust / fail-open.** If anything goes wrong (no IPC address, connection
  refused, malformed reply, timeout), it prints an empty JSON object ``{}`` and
  exits 0. An empty object is the CLI's "no opinion" hook output, so a broken
  shim degrades to *the CLI's normal behavior* (e.g. the TUI permission dialog),
  never a hang or a crash of the turn.

Wire protocol (newline-delimited JSON, one round-trip per connection):

    -> {"addr": "<token>", "event": <hook-event-json-from-cli>}\\n
    <- <hook-output-json>\\n

The ``addr`` token is an opaque shared secret (also passed in the env) so a
stray local process cannot trivially drive the SDK's callbacks. The SDK
validates it before dispatching.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys

# Environment variable carrying the IPC endpoint + auth token. Format:
#   "unix:<path>:<token>"   (Unix domain socket; preferred)
#   "tcp:<host>:<port>:<token>"  (TCP loopback fallback, e.g. Windows)
ENV_HOOK_IPC = "CLAUDE_AGENT_SDK_HOOK_IPC"

# A generous per-call timeout. The SDK side runs the user's callback which may
# itself be slow; the CLI's own hook timeout (default 10 min) is the real upper
# bound, so we just need to avoid hanging forever on a dead SDK.
_TIMEOUT_S = 600.0

# Cap reads so a runaway/garbage peer cannot exhaust memory.
_MAX_REPLY_BYTES = 8 * 1024 * 1024


def _emit_noop() -> None:
    """Print the CLI's 'no opinion' hook output and exit cleanly (fail-open)."""
    sys.stdout.write("{}")
    sys.stdout.flush()


def _connect(spec: str) -> tuple[socket.socket, str]:
    """Open a connection to the IPC endpoint described by ``spec``.

    Returns ``(socket, token)``. Raises on any failure (caller fails open).
    """
    if spec.startswith("unix:"):
        # unix:<path>:<token>  -- the path may itself contain ':' only on exotic
        # filesystems; we split off the token from the RIGHT to be safe.
        rest = spec[len("unix:") :]
        path, _, token = rest.rpartition(":")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(_TIMEOUT_S)
        sock.connect(path)
        return sock, token
    if spec.startswith("tcp:"):
        # tcp:<host>:<port>:<token>
        rest = spec[len("tcp:") :]
        head, _, token = rest.rpartition(":")
        host, _, port = head.rpartition(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(_TIMEOUT_S)
        sock.connect((host, int(port)))
        return sock, token
    raise ValueError(f"unrecognized hook IPC spec: {spec[:8]!r}")


def _read_line(sock: socket.socket) -> bytes:
    """Read one newline-terminated reply (bounded)."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk or total > _MAX_REPLY_BYTES:
            break
    return b"".join(chunks)


def main() -> int:
    # Read the hook event JSON the CLI wrote to our stdin.
    try:
        raw_in = sys.stdin.read()
    except Exception:
        _emit_noop()
        return 0

    spec = os.environ.get(ENV_HOOK_IPC)
    if not spec:
        # No IPC endpoint configured -> nothing to dispatch to. Fail open.
        _emit_noop()
        return 0

    try:
        event = json.loads(raw_in) if raw_in.strip() else {}
    except Exception:
        event = {"_raw": raw_in}

    sock: socket.socket | None = None
    try:
        sock, token = _connect(spec)
        request = json.dumps({"addr": token, "event": event})
        sock.sendall(request.encode("utf-8") + b"\n")
        # Half-close our write side so the SDK knows the request is complete even
        # if it reads to EOF rather than to the newline.
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_WR)
        reply = _read_line(sock)
    except Exception:
        _emit_noop()
        return 0
    finally:
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.close()

    text = reply.decode("utf-8", "replace").strip()
    if not text:
        _emit_noop()
        return 0
    # Validate it is JSON; if not, fail open rather than feed the CLI garbage
    # (the CLI treats non-JSON stdout as plain text additional context).
    try:
        json.loads(text)
    except Exception:
        _emit_noop()
        return 0
    sys.stdout.write(text)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
