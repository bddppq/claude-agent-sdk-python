"""An always-on, transport-internal, pure-relay API monitor.

The PTY transport tails the CLI's ``.jsonl`` transcript, which cannot show the
real CLI<->API exchange (per-call usage including helper-model calls, real API
timing, HTTP error statuses, the full ``tool_use.input`` while a permission
dialog is still blocking). This module fills that gap WITHOUT touching the
forwarded bytes:

* It runs a tiny loopback HTTP server bound to ``127.0.0.1:0`` (an ephemeral
  port). ``connect()`` points the CHILD CLI's ``ANTHROPIC_BASE_URL`` at it; the
  user never sees this.
* For each request it opens a connection to the real upstream (the original
  ``ANTHROPIC_BASE_URL``, default ``https://api.anthropic.com``) and forwards
  method / path / headers / body UNCHANGED, then streams the response back to
  the CLI UNCHANGED -- SSE chunks are relayed as they arrive (never buffered),
  so streaming still works for the CLI.
* It TEES a copy of ``/v1/messages`` request+response to an in-process callback
  for parsing only. The tee runs behind ``try/except`` and AFTER the bytes have
  already been forwarded, so a parsing error can never affect the relay or the
  turn.

This is ALWAYS ON. There is no config option, env toggle, or opt-out: it is
simply how the transport launches the CLI. The relay is a pure byte-for-byte
forwarder; only an out-of-band copy is parsed.
"""

from __future__ import annotations

import gzip
import json
import logging
import ssl
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any, cast
from urllib.parse import urlsplit

import anyio
from anyio.abc import ByteStream, SocketAttribute, SocketStream

from .._task_compat import TaskHandle, spawn_detached

logger = logging.getLogger(__name__)

# How many bytes to read per relay chunk. Small enough to relay SSE deltas
# promptly, large enough to be efficient for big request/response bodies.
_CHUNK = 65536

# Cap the tee'd copies so a pathological response cannot grow memory without
# bound. The monitor only needs the (small) JSON / SSE bodies of /v1/messages;
# anything larger is simply not parsed (the relay is unaffected).
_MAX_TEE_BYTES = 16 * 1024 * 1024

# Callback signature: (record: dict) -> None. record carries the parsed traffic
# for one /v1/messages call (see _parse_and_tee for the shape).
ApiCallback = Callable[[dict[str, Any]], None]


def _split_head(buffer: bytes) -> tuple[bytes, bytes] | None:
    """Split an HTTP message buffer at the end of its header block.

    Returns ``(head, rest)`` where ``head`` ends with the ``\\r\\n\\r\\n``
    terminator, or ``None`` if the full header block has not arrived yet.
    """
    idx = buffer.find(b"\r\n\r\n")
    if idx == -1:
        return None
    return buffer[: idx + 4], buffer[idx + 4 :]


def _parse_request_head(head: bytes) -> tuple[str, str, dict[str, str]]:
    """Parse a request head into (method, path, headers-lowercased)."""
    lines = head.split(b"\r\n")
    request_line = lines[0].decode("latin-1")
    parts = request_line.split(" ")
    method = parts[0] if parts else ""
    path = parts[1] if len(parts) > 1 else ""
    headers: dict[str, str] = {}
    for raw in lines[1:]:
        if not raw or b":" not in raw:
            continue
        name, _, value = raw.partition(b":")
        headers[name.decode("latin-1").strip().lower()] = value.decode(
            "latin-1"
        ).strip()
    return method, path, headers


def _rewrite_request_head(head: bytes, upstream_host: str, prefix: str) -> bytes:
    """Rewrite the request head's Host header (and path prefix) for the upstream.

    Everything else -- method, all other headers, and the body that follows --
    is preserved byte-for-byte. Only the ``Host`` line is replaced (the CLI sent
    the loopback ``127.0.0.1:<port>``, which the API rejects as a private IP) and
    the request target is prefixed if the upstream base url carried a path.
    """
    block, sep, _ = head.partition(b"\r\n\r\n")
    lines = block.split(b"\r\n")
    if not lines:
        return head
    # Optionally prefix the request target (rare: base url with a path).
    if prefix:
        first = lines[0].split(b" ")
        if len(first) >= 2 and not first[1].startswith(prefix.encode("latin-1")):
            first[1] = prefix.encode("latin-1") + first[1]
            lines[0] = b" ".join(first)
    out: list[bytes] = [lines[0]]
    replaced = False
    for raw in lines[1:]:
        if raw[:5].lower() == b"host:":
            out.append(b"Host: " + upstream_host.encode("latin-1"))
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        out.append(b"Host: " + upstream_host.encode("latin-1"))
    # ``sep`` is the b"\r\n\r\n" terminator (empty if the head had none).
    return b"\r\n".join(out) + (sep if sep else b"\r\n\r\n")


def _parse_status(head: bytes) -> int:
    """Parse the status code out of a response head."""
    try:
        status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        return int(status_line.split(" ")[1])
    except (IndexError, ValueError):
        return 0


def _content_length(headers: dict[str, str]) -> int | None:
    cl = headers.get("content-length")
    if cl is None:
        return None
    try:
        return int(cl)
    except ValueError:
        return None


def _maybe_gunzip(headers: dict[str, str], body: bytes) -> bytes:
    if "gzip" in headers.get("content-encoding", "").lower():
        try:
            return gzip.decompress(body)
        except (OSError, EOFError, ValueError):
            return body
    return body


def _dechunk(headers: dict[str, str], body: bytes) -> bytes:
    """Decode HTTP chunked transfer-encoding from a copy (for parsing only).

    Returns ``body`` unchanged when not chunked. Best-effort: a malformed frame
    just stops decoding and returns what was decoded so far (the relay path is
    unaffected -- this only touches the teed copy).
    """
    if "chunked" not in headers.get("transfer-encoding", "").lower():
        return body
    out = bytearray()
    pos = 0
    n = len(body)
    while pos < n:
        nl = body.find(b"\r\n", pos)
        if nl == -1:
            break
        try:
            size = int(bytes(body[pos:nl]).split(b";", 1)[0].strip(), 16)
        except ValueError:
            break
        start = nl + 2
        out.extend(body[start : start + size])
        pos = start + size + 2  # skip data + trailing CRLF
        if size == 0:
            break
    return bytes(out)


class ApiMonitor:
    """Loopback pure-relay proxy in front of the Anthropic API."""

    def __init__(self, upstream_base_url: str, callback: ApiCallback) -> None:
        split = urlsplit(upstream_base_url)
        self._upstream_host = split.hostname or "api.anthropic.com"
        self._upstream_tls = (split.scheme or "https") == "https"
        self._upstream_port = split.port or (443 if self._upstream_tls else 80)
        # A non-empty upstream path prefix (rare for a base url) is preserved.
        self._upstream_prefix = split.path.rstrip("/")
        self._callback = callback
        self._listener: Any = None
        self._serve_handle: TaskHandle | None = None
        self._port: int | None = None
        self._ssl_context: ssl.SSLContext | None = (
            ssl.create_default_context() if self._upstream_tls else None
        )

    @property
    def port(self) -> int:
        if self._port is None:
            raise RuntimeError("ApiMonitor.start() has not completed")
        return self._port

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        """Bind the loopback listener and start serving in the background.

        The serve loop is a *detached* task (via ``spawn_detached``) rather than
        a held-open anyio task group: a task group's cancel scope has task
        affinity, so tearing it down from a different task than it was entered in
        (which ``close()`` can do) raises ``RuntimeError``. ``spawn_detached``
        returns a handle that is safe to ``cancel()`` from any task -- the same
        reason the PTY transport uses it for its loops.
        """
        self._listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
        # Record the actual ephemeral port chosen by the OS.
        for listener in getattr(self._listener, "listeners", [self._listener]):
            raw = listener.extra(SocketAttribute.local_address)
            self._port = raw[1]
            break
        self._serve_handle = spawn_detached(self._serve())

    async def _serve(self) -> None:
        assert self._listener is not None
        try:
            await self._listener.serve(self._handle_client)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            logger.debug("API monitor serve loop ended", exc_info=True)

    async def stop(self) -> None:
        """Stop serving and release the listener (safe from any task)."""
        if self._serve_handle is not None:
            self._serve_handle.cancel()
            with suppress(Exception):
                await self._serve_handle.wait()
            self._serve_handle = None
        if self._listener is not None:
            with suppress(Exception):
                await self._listener.aclose()
            self._listener = None

    # ------------------------------------------------------------------ #
    # Per-connection relay
    # ------------------------------------------------------------------ #

    async def _handle_client(self, client: SocketStream) -> None:
        """Relay one client connection to the upstream, then tee a copy."""
        try:
            async with client:
                await self._relay_connection(client)
        except anyio.get_cancelled_exc_class():
            raise
        except Exception:
            # A relay error must never crash the monitor; the worst case is the
            # CLI sees a dropped connection and retries (which we also observe).
            logger.debug("API monitor connection relay failed", exc_info=True)

    async def _relay_connection(self, client: SocketStream) -> None:
        # Read the request head so we know the path/headers and how to read the
        # body. Keep the connection one-request-per-connection: the CLI's HTTP
        # client opens a fresh connection per call in practice, and closing
        # after the response is a valid HTTP/1.1 behavior we signal via the
        # forwarded headers being passed through unchanged.
        head, rest = await self._read_head(client)
        if head is None:
            return
        method, path, req_headers = _parse_request_head(head)

        upstream = await self._connect_upstream()
        # Real per-call API duration: from just before we forward the request to
        # just after the full response has been relayed back (R7).
        started = time.monotonic()
        try:
            async with upstream:
                req_body = await self._relay_request(
                    client, upstream, head, rest, req_headers
                )
                status, resp_headers, resp_body = await self._relay_response(
                    upstream, client
                )
            duration_ms = int((time.monotonic() - started) * 1000)
            self._safe_tee(
                method,
                path,
                req_headers,
                req_body,
                status,
                resp_headers,
                resp_body,
                duration_ms,
            )
        finally:
            pass

    async def _connect_upstream(self) -> ByteStream:
        # ``tls`` is a runtime bool, so mypy cannot pick the right connect_tcp
        # overload (SocketStream vs TLSStream); both satisfy ByteStream and the
        # relay treats them identically (send / receive / aclose), so ignore the
        # overload selection. tls_standard_compatible=False tolerates an upstream
        # that closes the TCP connection without a TLS close_notify (common for
        # SSE).
        stream = await anyio.connect_tcp(  # type: ignore[call-overload]
            self._upstream_host,
            self._upstream_port,
            tls=self._upstream_tls,
            ssl_context=self._ssl_context,
            tls_standard_compatible=False,
        )
        return cast(ByteStream, stream)

    async def _read_head(self, stream: ByteStream) -> tuple[bytes | None, bytes]:
        buffer = b""
        while True:
            split = _split_head(buffer)
            if split is not None:
                return split
            try:
                chunk = await stream.receive(_CHUNK)
            except (anyio.EndOfStream, anyio.ClosedResourceError):
                return None, b""
            if not chunk:
                return None, b""
            buffer += chunk

    async def _relay_request(
        self,
        client: SocketStream,
        upstream: ByteStream,
        head: bytes,
        rest: bytes,
        headers: dict[str, str],
    ) -> bytes:
        """Forward request head+body to the upstream; return a body copy.

        The body and every header are forwarded byte-for-byte EXCEPT the ``Host``
        header, which is rewritten from the loopback ``127.0.0.1:<port>`` to the
        real upstream host. This is mandatory framing, not content alteration:
        the upstream (and Anthropic's edge) rejects a request whose Host resolves
        to a private/reserved IP (``403 ip_authority_private``). The request
        target path may also be prefixed if the upstream base url had a path.
        """
        out_head = _rewrite_request_head(
            head, self._upstream_host, self._upstream_prefix
        )
        await upstream.send(out_head)
        body = bytearray()
        if rest:
            await upstream.send(rest)
            body += rest

        remaining = self._body_remaining(headers, len(rest))
        # Read and forward the rest of the body. Bytes are forwarded the instant
        # they arrive; the copy (for tee) is bounded.
        while remaining != 0:
            try:
                chunk = await client.receive(_CHUNK)
            except (anyio.EndOfStream, anyio.ClosedResourceError):
                break
            if not chunk:
                break
            await upstream.send(chunk)
            if len(body) < _MAX_TEE_BYTES:
                body += chunk
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
        return bytes(body)

    def _body_remaining(self, headers: dict[str, str], already: int) -> int | None:
        """Bytes of request body still to read, or None for read-to-EOF.

        Returns 0 when the framing says the whole body is already in hand.
        """
        cl = _content_length(headers)
        if cl is not None:
            return max(cl - already, 0)
        if "chunked" in headers.get("transfer-encoding", "").lower():
            return None
        # No body framing (e.g. GET): nothing more to read.
        return 0

    async def _relay_response(
        self, upstream: ByteStream, client: SocketStream
    ) -> tuple[int, dict[str, str], bytes]:
        """Stream the response back to the client UNCHANGED; return a copy.

        Reads the head first to learn the framing, but forwards every byte to
        the client the instant it arrives so SSE streaming is preserved.
        """
        head, rest = await self._read_head(upstream)
        if head is None:
            return 0, {}, b""
        await client.send(head)
        status = _parse_status(head)
        resp_headers = _parse_headers(head)

        body = bytearray()

        async def _emit(chunk: bytes) -> None:
            # Forward FIRST so the relay is never delayed (SSE chunks reach the
            # CLI as they arrive); copy a bounded amount for the tee.
            await client.send(chunk)
            if len(body) < _MAX_TEE_BYTES:
                body.extend(chunk)

        # Relay the body honoring its framing. A keep-alive Content-Length
        # response does NOT EOF, so reading-until-close would hang forever -- we
        # must stop at Content-Length / the terminal chunk. Only a response with
        # neither framing (close-delimited, e.g. some SSE) reads until EOF.
        await self._relay_response_body(upstream, resp_headers, rest, _emit)
        return status, resp_headers, bytes(body)

    async def _relay_response_body(
        self,
        upstream: ByteStream,
        headers: dict[str, str],
        leftover: bytes,
        emit: Callable[[bytes], Any],
    ) -> None:
        """Forward the response body to ``emit`` per its HTTP framing."""
        te = headers.get("transfer-encoding", "").lower()
        cl = _content_length(headers)

        if "chunked" in te:
            await self._relay_chunked_body(upstream, leftover, emit)
            return

        if cl is not None:
            sent = 0
            if leftover:
                take = leftover[:cl]
                await emit(take)
                sent += len(take)
            while sent < cl:
                try:
                    chunk = await upstream.receive(min(_CHUNK, cl - sent))
                except (anyio.EndOfStream, anyio.ClosedResourceError):
                    break
                if not chunk:
                    break
                await emit(chunk)
                sent += len(chunk)
            return

        # No Content-Length and not chunked: close-delimited body (read to EOF).
        if leftover:
            await emit(leftover)
        while True:
            try:
                chunk = await upstream.receive(_CHUNK)
            except (anyio.EndOfStream, anyio.ClosedResourceError):
                break
            if not chunk:
                break
            await emit(chunk)

    async def _relay_chunked_body(
        self,
        upstream: ByteStream,
        leftover: bytes,
        emit: Callable[[bytes], Any],
    ) -> None:
        """Relay a chunked body verbatim, stopping at the terminal 0-size chunk.

        Forwards the RAW chunk framing (sizes + CRLFs) unchanged so the client
        sees byte-identical chunked encoding, but parses just enough to know when
        the body ends -- otherwise a keep-alive connection never EOFs and the
        relay hangs.
        """
        buf = bytearray(leftover)

        async def _more() -> bool:
            try:
                data = await upstream.receive(_CHUNK)
            except (anyio.EndOfStream, anyio.ClosedResourceError):
                return False
            if not data:
                return False
            buf.extend(data)
            return True

        while True:
            nl = buf.find(b"\r\n")
            while nl == -1:
                if not await _more():
                    if buf:
                        await emit(bytes(buf))
                    return
                nl = buf.find(b"\r\n")
            try:
                size = int(bytes(buf[:nl]).split(b";", 1)[0].strip(), 16)
            except ValueError:
                # Malformed framing: relay whatever remains and stop.
                await emit(bytes(buf))
                return
            frame_end = nl + 2 + size + 2  # size line CRLF + data + trailing CRLF
            while len(buf) < frame_end:
                if not await _more():
                    if buf:
                        await emit(bytes(buf))
                    return
            await emit(bytes(buf[:frame_end]))
            del buf[:frame_end]
            if size == 0:
                return

    # ------------------------------------------------------------------ #
    # Tee (parse a copy; never affects the relay)
    # ------------------------------------------------------------------ #

    def _safe_tee(
        self,
        method: str,
        path: str,
        req_headers: dict[str, str],
        req_body: bytes,
        status: int,
        resp_headers: dict[str, str],
        resp_body: bytes,
        duration_ms: int = 0,
    ) -> None:
        # Forward-first contract: this runs AFTER the bytes were relayed, and any
        # error here is swallowed so it can never affect the turn.
        try:
            self._parse_and_tee(
                method,
                path,
                req_headers,
                req_body,
                status,
                resp_headers,
                resp_body,
                duration_ms,
            )
        except Exception:
            logger.debug("API monitor tee parse failed", exc_info=True)

    def _parse_and_tee(
        self,
        method: str,
        path: str,
        req_headers: dict[str, str],
        req_body: bytes,
        status: int,
        resp_headers: dict[str, str],
        resp_body: bytes,
        duration_ms: int = 0,
    ) -> None:
        # Only /v1/messages carries the usage/tool/streaming data we enrich from.
        if "/v1/messages" not in path:
            return

        request_json: dict[str, Any] | None = None
        decoded_req = _maybe_gunzip(req_headers, _dechunk(req_headers, req_body))
        if decoded_req:
            try:
                parsed = json.loads(decoded_req)
                if isinstance(parsed, dict):
                    request_json = parsed
            except (json.JSONDecodeError, ValueError):
                request_json = None

        # The CLI's responses are gzip-encoded AND chunked-framed -- de-chunk the
        # teed COPY before gunzip+parse (the relayed bytes were forwarded raw and
        # are unaffected by this parse-only transformation).
        decoded_resp = _maybe_gunzip(resp_headers, _dechunk(resp_headers, resp_body))
        usage, stop_reason, content_blocks, partial_text, model = _parse_response_body(
            resp_headers, decoded_resp
        )
        # Fall back to the request's model when the response did not carry one
        # (e.g. an error response): the per-call cost is keyed on the model.
        if model is None and isinstance(request_json, dict):
            req_model = request_json.get("model")
            if isinstance(req_model, str):
                model = req_model

        record: dict[str, Any] = {
            "path": path,
            "method": method,
            "status": status,
            "duration_ms": duration_ms,
            "model": model,
            "request": request_json,
            "usage": usage,
            "stop_reason": stop_reason,
            "content_blocks": content_blocks,
            "partial_text": partial_text,
        }
        self._callback(record)


def _parse_headers(head: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in head.split(b"\r\n")[1:]:
        if not raw or b":" not in raw:
            continue
        name, _, value = raw.partition(b":")
        headers[name.decode("latin-1").strip().lower()] = value.decode(
            "latin-1"
        ).strip()
    return headers


def _parse_response_body(
    headers: dict[str, str], body: bytes
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]], str, str | None]:
    """Parse a /v1/messages response into (usage, stop_reason, blocks, text, model).

    Handles both a plain JSON ``message`` response and an SSE
    ``text/event-stream`` (the CLI's normal mode), reconstructing the final
    usage / stop_reason / content blocks / concatenated text delta, and the
    response model id.
    """
    if not body:
        return None, None, [], "", None
    content_type = headers.get("content-type", "").lower()
    text = body.decode("utf-8", "replace")

    if "text/event-stream" in content_type or text.lstrip().startswith("event:"):
        return _parse_sse(text)

    # Plain JSON message response.
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None, None, [], "", None
    if not isinstance(obj, dict):
        return None, None, [], "", None
    raw_usage = obj.get("usage")
    usage: dict[str, Any] | None = raw_usage if isinstance(raw_usage, dict) else None
    raw_stop = obj.get("stop_reason")
    stop_reason: str | None = raw_stop if isinstance(raw_stop, str) else None
    raw_content = obj.get("content")
    blocks: list[dict[str, Any]] = (
        [b for b in raw_content if isinstance(b, dict)]
        if isinstance(raw_content, list)
        else []
    )
    partial = "".join(
        b.get("text", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )
    raw_model = obj.get("model")
    model: str | None = raw_model if isinstance(raw_model, str) else None
    return usage, stop_reason, blocks, partial, model


def _parse_sse(
    text: str,
) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]], str, str | None]:
    """Reconstruct the final state from an SSE /v1/messages stream.

    Anthropic streaming emits ``message_start`` (initial usage + model), per-block
    ``content_block_start`` / ``content_block_delta`` / ``content_block_stop``,
    and ``message_delta`` (final usage + stop_reason). We merge the usage from
    message_start and message_delta and rebuild the content blocks.
    """
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    blocks: dict[int, dict[str, Any]] = {}
    text_parts: list[str] = []
    model: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        etype = event.get("type")

        if etype == "message_start":
            msg = event.get("message")
            if isinstance(msg, dict):
                if isinstance(msg.get("usage"), dict):
                    usage.update(msg["usage"])
                if isinstance(msg.get("model"), str):
                    model = msg["model"]
        elif etype == "message_delta":
            delta_usage = event.get("usage")
            if isinstance(delta_usage, dict):
                usage.update(delta_usage)
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
        elif etype == "content_block_start":
            idx = event.get("index")
            block = event.get("content_block")
            if isinstance(idx, int) and isinstance(block, dict):
                blocks[idx] = dict(block)
        elif etype == "content_block_delta":
            idx = event.get("index")
            delta = event.get("delta")
            if isinstance(idx, int) and isinstance(delta, dict):
                block = blocks.setdefault(idx, {})
                dtype = delta.get("type")
                if dtype == "text_delta":
                    chunk = delta.get("text", "")
                    block["text"] = block.get("text", "") + chunk
                    text_parts.append(chunk)
                elif dtype == "input_json_delta":
                    block["partial_json"] = block.get("partial_json", "") + delta.get(
                        "partial_json", ""
                    )
                elif dtype == "thinking_delta":
                    block["thinking"] = block.get("thinking", "") + delta.get(
                        "thinking", ""
                    )

    # Finalize tool_use blocks: parse the accumulated partial_json into input.
    ordered = [blocks[i] for i in sorted(blocks)]
    for block in ordered:
        if block.get("type") == "tool_use" and "partial_json" in block:
            raw = block.pop("partial_json")
            try:
                block["input"] = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, ValueError):
                block.setdefault("input", {})

    return (usage or None), stop_reason, ordered, "".join(text_parts), model
