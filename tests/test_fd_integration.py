"""End-to-end regression test for the file-descriptor leak (issue #7).

Unlike the unit tests in ``test_translator.py`` (which patch ``Agent``), this
test drives the *real* pydantic-ai stack against a local OpenAI-compatible stub
server and counts the TCP connections the server accepts.

Each pydantic-ai ``Agent`` builds its own, un-cached ``httpx.AsyncClient``
(``OpenAIProvider`` calls ``create_async_http_client()`` per instance), and each
client opens one keep-alive connection that is reused for every request it makes.
So the number of accepted connections is a direct, GC-independent measure of how
many clients were created:

* Before the fix (a new ``Agent`` per batch) the server sees one connection per
  batch -- i.e. ``num_batches`` sockets, which is the ``[Errno 24]`` leak.
* After the fix (agents reused via the per-thread, per-language cache) the server
  sees at most one connection per worker thread.

No API key or network access is needed; everything runs against localhost.
"""

import json
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from xcstrings_translator.models import (
    Localization,
    StringEntry,
    StringUnit,
    XCStringsFile,
)
from xcstrings_translator.translator import XCStringsTranslator


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI chat-completions stub that forces a tool-call response.

    ``connection_count`` is incremented once per accepted TCP connection
    (``setup`` runs once per connection, not per request), so HTTP keep-alive
    means a reused client counts only once no matter how many batches it sends.
    """

    protocol_version = "HTTP/1.1"
    connection_count = 0
    _lock = threading.Lock()

    def setup(self):
        with _StubHandler._lock:
            _StubHandler.connection_count += 1
        super().setup()

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        # pydantic-ai requests structured output via a forced tool call; echo the
        # tool name back with empty translations (the test asserts on connection
        # count, not on translation content).
        tool_name = body["tools"][0]["function"]["name"]
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({"translations": []}),
                    },
                }
            ],
        }
        payload = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub",
                "choices": [
                    {"index": 0, "finish_reason": "tool_calls", "message": message}
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.mark.integration
def test_agents_reuse_connections_no_fd_leak(monkeypatch):
    """Many batches must not open one socket per batch (regression for #7)."""
    _StubHandler.connection_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Route the real OpenAI client at our local stub (honoured by pydantic-ai's
    # OpenAIProvider -> AsyncOpenAI).
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    num_batches = 60
    concurrency = 4
    xc = XCStringsFile(
        sourceLanguage="en",
        strings={
            f"Key{i}": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value=f"Key{i}"))
                }
            )
            for i in range(num_batches)
        },
    )
    translator = XCStringsTranslator(
        model="openai-chat:gpt-4o-mini", batch_size=1, concurrency=concurrency
    )
    try:
        with warnings.catch_warnings():
            # Silence any provider/SDK warnings emitted during the real run; the
            # test asserts on connection count, not on warning output.
            warnings.simplefilter("ignore")
            translator.translate_file(xc, ["fr"])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    connections = _StubHandler.connection_count
    # Agents are reused, so at most one keep-alive connection per worker thread.
    # Pre-fix this would equal num_batches (one httpx client per batch).
    assert connections <= concurrency, (
        f"expected <= {concurrency} connections (one per worker thread), got "
        f"{connections} for {num_batches} batches -- a per-batch httpx client "
        f"would open one socket per batch (the [Errno 24] leak)"
    )
