"""An app that holds `/stream` responses open until told to finish them."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from urllib.parse import parse_qs

if TYPE_CHECKING:
    from asgiref.typing import ASGIReceiveCallable, ASGISendCallable, HTTPScope, Scope

_POLL_INTERVAL = 0.005

_next_ticket = 0
_released = 0


async def _respond(send: ASGISendCallable, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
            "trailers": False,
        }
    )
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _stream(receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
    global _next_ticket  # noqa: PLW0603
    ticket = _next_ticket
    _next_ticket += 1
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
            "trailers": False,
        }
    )
    await send({"type": "http.response.body", "body": b"", "more_body": True})
    disconnect = asyncio.ensure_future(receive())
    try:
        while _released <= ticket:
            if disconnect.done() and disconnect.result()["type"] == "http.disconnect":
                return
            await asyncio.sleep(_POLL_INTERVAL)
    finally:
        disconnect.cancel()
    await send({"type": "http.response.body", "body": b"done", "more_body": False})


async def _release(scope: HTTPScope, send: ASGISendCallable) -> None:
    global _released  # noqa: PLW0603
    query = parse_qs(scope["query_string"].decode())
    if query.get("all"):
        _released = _next_ticket
    else:
        _released = min(_next_ticket, _released + int(query.get("n", ["1"])[0]))
    await _respond(send, str(_released).encode())


async def app(
    scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable
) -> None:
    if scope["type"] != "http":
        msg = f"Unsupported scope type: {scope['type']}"
        raise RuntimeError(msg)
    match scope["path"]:
        case "/stream":
            await _stream(receive, send)
        case "/release":
            await _release(scope, send)
        case _:
            await _respond(send, b"hello")
