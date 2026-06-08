"""Unit tests for the always-on pure-relay API monitor.

These run a fake upstream HTTP server in-process, point an :class:`ApiMonitor`
at it, drive a client through the monitor, and assert:

* the request reaches the upstream UNCHANGED (pure relay), and the response
  reaches the client UNCHANGED (byte-for-byte);
* SSE responses are relayed as a stream (not buffered) and the tee reconstructs
  the per-call usage / stop_reason / content from the event stream;
* a gzipped JSON response body is gunzipped for the parse copy only;
* the full ``tool_use.input`` is recovered from the response (the RV2 signal).

No real network, no subprocess, no model.
"""

import gzip
import json

import anyio
import pytest
from anyio.abc import SocketAttribute

from claude_agent_sdk._internal.transport._api_monitor import (
    ApiMonitor,
    _parse_sse,
)

pytestmark = pytest.mark.anyio


def _listener_port(listener: object) -> int:
    for sub in getattr(listener, "listeners", [listener]):
        return int(sub.extra(SocketAttribute.local_address)[1])  # type: ignore[attr-defined]
    raise AssertionError("no listener port")


class FakeUpstream:
    """A minimal HTTP/1.1 upstream that returns a canned response.

    Records the exact request head + body it received so a test can assert the
    relay forwarded them unchanged.
    """

    def __init__(self, response: bytes) -> None:
        self._response = response
        self.requests: list[tuple[bytes, bytes]] = []
        self.listener: object = None
        self.port = 0

    async def start(self, tg: anyio.abc.TaskGroup) -> None:
        self.listener = await anyio.create_tcp_listener(local_host="127.0.0.1")
        self.port = _listener_port(self.listener)
        tg.start_soon(self.listener.serve, self._handle)  # type: ignore[attr-defined]

    async def _handle(self, conn: anyio.abc.SocketStream) -> None:
        async with conn:
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = await conn.receive(4096)
                if not chunk:
                    return
                buf += chunk
            head, rest = buf.split(b"\r\n\r\n", 1)
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            body = rest
            while len(body) < length:
                body += await conn.receive(4096)
            self.requests.append((head, body))
            await conn.send(self._response)


async def _drive(
    monitor_upstream_url_factory,  # noqa: ANN001 - test helper
    response: bytes,
    request: bytes,
    callback,  # noqa: ANN001
) -> bytes:
    """Run upstream + monitor, send one request through, return the relayed bytes."""
    upstream = FakeUpstream(response)
    async with anyio.create_task_group() as tg:
        await upstream.start(tg)
        monitor = ApiMonitor(monitor_upstream_url_factory(upstream.port), callback)
        await monitor.start()
        try:
            client = await anyio.connect_tcp("127.0.0.1", monitor.port)
            await client.send(request)
            got = b""
            try:
                while True:
                    chunk = await client.receive(4096)
                    if not chunk:
                        break
                    got += chunk
            except anyio.EndOfStream:
                pass
            await client.aclose()
            # Give the post-forward tee a moment to run (it runs after the
            # response is fully relayed, on the same connection task).
            await anyio.sleep(0.05)
        finally:
            await monitor.stop()
            tg.cancel_scope.cancel()
    _drive.last_upstream = upstream  # type: ignore[attr-defined]
    return got


SSE_BODY = (
    b"event: message_start\r\n"
    b'data: {"type":"message_start","message":{"id":"msg_1",'
    b'"model":"claude-opus-4-8","content":[],'
    b'"usage":{"input_tokens":10,"cache_read_input_tokens":100,"output_tokens":1}}}\r\n\r\n'
    b"event: content_block_start\r\n"
    b'data: {"type":"content_block_start","index":0,'
    b'"content_block":{"type":"text","text":""}}\r\n\r\n'
    b"event: content_block_delta\r\n"
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"PONG"}}\r\n\r\n'
    b"event: content_block_stop\r\n"
    b'data: {"type":"content_block_stop","index":0}\r\n\r\n'
    b"event: message_delta\r\n"
    b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    b'"usage":{"output_tokens":5}}\r\n\r\n'
    b"event: message_stop\r\n"
    b'data: {"type":"message_stop"}\r\n\r\n'
)


def _sse_response() -> bytes:
    return (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Content-Length: " + str(len(SSE_BODY)).encode() + b"\r\n\r\n" + SSE_BODY
    )


def _messages_request(body: bytes) -> bytes:
    return (
        b"POST /v1/messages HTTP/1.1\r\n"
        b"Host: original.example\r\n"
        b"Content-Type: application/json\r\n"
        b"X-Api-Key: secret-123\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )


class TestRelayTransparency:
    async def test_request_and_response_relayed_unchanged(self):
        teed: list[dict] = []
        body = (
            b'{"model":"claude-opus-4-8","messages":[{"role":"user","content":"hi"}]}'
        )
        got = await _drive(
            lambda port: f"http://127.0.0.1:{port}",
            _sse_response(),
            _messages_request(body),
            teed.append,
        )
        upstream = _drive.last_upstream  # type: ignore[attr-defined]

        # Response reached the client byte-for-byte.
        assert got == _sse_response()

        # Upstream saw the request body UNCHANGED (pure relay -- never altered).
        up_head, up_body = upstream.requests[0]
        assert up_body == body
        # Auth header forwarded unchanged.
        assert b"X-Api-Key: secret-123" in up_head
        # The forwarded head preserves the method/path.
        assert up_head.split(b"\r\n", 1)[0] == b"POST /v1/messages HTTP/1.1"

    async def test_sse_tee_reconstructs_usage_and_text(self):
        teed: list[dict] = []
        body = b'{"model":"claude-opus-4-8","messages":[]}'
        await _drive(
            lambda port: f"http://127.0.0.1:{port}",
            _sse_response(),
            _messages_request(body),
            teed.append,
        )
        assert len(teed) == 1
        record = teed[0]
        assert record["status"] == 200
        assert record["path"] == "/v1/messages"
        # usage merged from message_start + message_delta.
        assert record["usage"]["input_tokens"] == 10
        assert record["usage"]["cache_read_input_tokens"] == 100
        assert record["usage"]["output_tokens"] == 5
        assert record["stop_reason"] == "end_turn"
        assert record["partial_text"] == "PONG"
        # Request JSON parsed.
        assert record["request"]["model"] == "claude-opus-4-8"
        # Model surfaced from the response, and real per-call timing captured.
        assert record["model"] == "claude-opus-4-8"
        assert "duration_ms" in record and record["duration_ms"] >= 0


class TestGzipJsonResponse:
    async def test_gzipped_json_body_is_gunzipped_for_parse(self):
        message = {
            "id": "msg_2",
            "model": "claude-opus-4-8",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        }
        raw = gzip.compress(json.dumps(message).encode())
        response = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Encoding: gzip\r\n"
            b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw
        )
        teed: list[dict] = []
        body = b'{"model":"claude-opus-4-8","messages":[]}'
        got = await _drive(
            lambda port: f"http://127.0.0.1:{port}",
            response,
            _messages_request(body),
            teed.append,
        )
        # Relayed bytes are still the gzipped original (never altered).
        assert got == response
        # The tee gunzipped a COPY only and parsed it.
        assert teed[0]["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert teed[0]["stop_reason"] == "end_turn"
        assert teed[0]["partial_text"] == "hello"


class TestToolUseInputRecovered:
    async def test_full_tool_use_input_from_sse(self):
        sse = (
            b"event: message_start\r\n"
            b'data: {"type":"message_start","message":{"id":"m","model":'
            b'"claude-opus-4-8","content":[],"usage":{"input_tokens":1,"output_tokens":1}}}\r\n\r\n'
            b"event: content_block_start\r\n"
            b'data: {"type":"content_block_start","index":0,"content_block":'
            b'{"type":"tool_use","id":"toolu_1","name":"Write","input":{}}}\r\n\r\n'
            b"event: content_block_delta\r\n"
            b'data: {"type":"content_block_delta","index":0,"delta":'
            b'{"type":"input_json_delta","partial_json":"{\\"file_path\\":\\"/tmp/x.txt\\","}}\r\n\r\n'
            b"event: content_block_delta\r\n"
            b'data: {"type":"content_block_delta","index":0,"delta":'
            b'{"type":"input_json_delta","partial_json":"\\"content\\":\\"hi\\"}"}}\r\n\r\n'
            b"event: content_block_stop\r\n"
            b'data: {"type":"content_block_stop","index":0}\r\n\r\n'
            b"event: message_delta\r\n"
            b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
            b'"usage":{"output_tokens":9}}\r\n\r\n'
        )
        response = (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Content-Length: " + str(len(sse)).encode() + b"\r\n\r\n" + sse
        )
        teed: list[dict] = []
        await _drive(
            lambda port: f"http://127.0.0.1:{port}",
            response,
            _messages_request(b'{"model":"claude-opus-4-8","messages":[]}'),
            teed.append,
        )
        blocks = teed[0]["content_blocks"]
        tool_blocks = [b for b in blocks if b.get("type") == "tool_use"]
        assert len(tool_blocks) == 1
        # The FULL original input is reconstructed from the input_json_delta
        # stream -- this is the RV2 signal (vs the TUI's scraped {target}).
        assert tool_blocks[0]["input"] == {
            "file_path": "/tmp/x.txt",
            "content": "hi",
        }
        assert tool_blocks[0]["name"] == "Write"
        assert tool_blocks[0]["id"] == "toolu_1"


class TestNonMessagesNotTeed:
    async def test_non_messages_path_is_relayed_but_not_teed(self):
        message = {"models": []}
        raw = json.dumps(message).encode()
        response = (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Content-Length: " + str(len(raw)).encode() + b"\r\n\r\n" + raw
        )
        request = b"GET /v1/models HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n"
        teed: list[dict] = []
        got = await _drive(
            lambda port: f"http://127.0.0.1:{port}",
            response,
            request,
            teed.append,
        )
        assert got == response
        # Only /v1/messages is teed; /v1/models is relayed silently.
        assert teed == []


class TestSseParser:
    """Direct unit tests of the SSE reconstruction (no sockets)."""

    def test_merges_usage_and_extracts_text(self):
        text = SSE_BODY.decode()
        usage, stop, blocks, partial, model = _parse_sse(text)
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5
        assert stop == "end_turn"
        assert partial == "PONG"
        assert blocks[0]["text"] == "PONG"
        assert model == "claude-opus-4-8"
