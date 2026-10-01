from __future__ import annotations

import asyncio
import gc
import threading
from typing import TYPE_CHECKING

import anyio
import pytest
from anyio import to_thread

from pyqwest import (
    HTTPTransport,
    HTTPVersion,
    Request,
    Response,
    SyncHTTPTransport,
    SyncRequest,
    SyncResponse,
)

from ._pool_server import PoolTestServer, free_port, settle, wait_for

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

STREAM_LIMIT = 2
"""The concurrent streams each server allows per connection."""

HOST = "pool.test"
"""The host name the DNS tests point at the servers."""


def _run_server() -> Iterator[PoolTestServer]:
    server = PoolTestServer(max_concurrent_streams=STREAM_LIMIT)
    # pyvoy drives its Envoy subprocess with asyncio. A private loop keeps the
    # server independent of the backend the async tests run on.
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(server.start())
        try:
            yield server
        finally:
            loop.run_until_complete(server.stop())
    finally:
        loop.close()


@pytest.fixture(scope="module")
def server() -> Iterator[PoolTestServer]:
    yield from _run_server()


@pytest.fixture(scope="module")
def other_server() -> Iterator[PoolTestServer]:
    yield from _run_server()


@pytest.fixture(autouse=True)
def _release_everything(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    for name in ("server", "other_server"):
        if name in request.fixturenames:
            request.getfixturevalue(name).release_all()


async def open_stream(transport: HTTPTransport, url: str) -> Response:
    return await transport.execute(Request("GET", f"{url}/stream"))


async def drain(response: Response) -> None:
    async for _ in response.content:
        pass


async def get(transport: HTTPTransport, url: str) -> None:
    await drain(await transport.execute(Request("GET", url)))


async def wait(getter: Callable[[], int], expected: int, what: str) -> None:
    await to_thread.run_sync(lambda: wait_for(getter, expected, what=what))


@pytest.mark.anyio
async def test_opens_connections_as_streams_fill(server: PoolTestServer) -> None:
    connections = server.connections()
    streams = server.streams()
    async with HTTPTransport(http_version=HTTPVersion.HTTP2) as transport:
        held = [await open_stream(transport, server.url) for _ in range(STREAM_LIMIT)]
        await wait(server.connections, connections + 1, "connections")

        extra: list[Response] = []

        async def open_extra() -> None:
            extra.append(await open_stream(transport, server.url))

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(open_extra)
            # Two more connections: the server allows 2 streams on each.
            await wait(server.streams, streams + STREAM_LIMIT + 3, "streams")
        assert server.connections() == connections + 3

        server.release_all()
        for response in [*held, *extra]:
            await drain(response)

        # Short requests reuse the connections now that their streams ended.
        for _ in range(5):
            await get(transport, f"{server.url}/short")
        assert server.connections() == connections + 3


def test_sync_opens_connections_as_streams_fill(server: PoolTestServer) -> None:
    connections = server.connections()
    streams = server.streams()
    with SyncHTTPTransport(http_version=HTTPVersion.HTTP2) as transport:
        held = [
            transport.execute_sync(SyncRequest("GET", f"{server.url}/stream"))
            for _ in range(STREAM_LIMIT)
        ]
        extra: list[SyncResponse] = []

        def open_extra() -> None:
            extra.append(
                transport.execute_sync(SyncRequest("GET", f"{server.url}/stream"))
            )

        threads = [threading.Thread(target=open_extra) for _ in range(3)]
        for thread in threads:
            thread.start()
        wait_for(server.streams, streams + STREAM_LIMIT + 3, what="streams")
        for thread in threads:
            thread.join(timeout=5)
        assert server.connections() == connections + 3

        server.release_all()
        for response in [*held, *extra]:
            for _ in response.content:
                pass


@pytest.mark.anyio
async def test_max_connections_per_address_queues_at_cap(
    server: PoolTestServer,
) -> None:
    connections = server.connections()
    streams = server.streams()
    async with HTTPTransport(
        http_version=HTTPVersion.HTTP2, max_connections_per_address=2
    ) as transport:
        held = [await open_stream(transport, server.url) for _ in range(STREAM_LIMIT)]
        extra: list[Response] = []

        async def open_extra() -> None:
            extra.append(await open_stream(transport, server.url))

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(open_extra)
            # One more connection takes two streams; the third waits on it.
            await wait(server.streams, streams + 2 * STREAM_LIMIT, "streams")
            assert await to_thread.run_sync(settle, server.streams) == streams + 4
            assert server.connections() == connections + 2

            # Ending a stream lets the waiting one start, still on 2 connections.
            server.release(1)
            await drain(held.pop(0))
            await wait(server.streams, streams + 5, "streams")
            assert server.connections() == connections + 2

        server.release_all()
        for response in [*held, *extra]:
            await drain(response)


@pytest.mark.anyio
async def test_max_connections_per_address_http1_waits_at_cap(
    server: PoolTestServer,
) -> None:
    """Each HTTP/1 stream takes a whole connection; past the cap a request waits
    for one to be returned rather than opening another."""
    connections = server.http1_connections()
    requests = server.http1_requests()
    async with HTTPTransport(
        http_version=HTTPVersion.HTTP1, max_connections_per_address=2
    ) as transport:
        held = [await open_stream(transport, server.url) for _ in range(2)]
        await wait(server.http1_connections, connections + 2, "connections")
        extra: list[Response] = []

        async def open_extra() -> None:
            extra.append(await open_stream(transport, server.url))

        async with anyio.create_task_group() as tg:
            tg.start_soon(open_extra)
            assert (
                await to_thread.run_sync(settle, server.http1_requests) == requests + 2
            )
            assert server.http1_connections() == connections + 2

            # Returning a connection lets the waiting request use it.
            await to_thread.run_sync(lambda: server.release(1, http2=True))
            await drain(held.pop(0))
            await wait(server.http1_requests, requests + 3, "requests")
            assert server.http1_connections() == connections + 2

        await to_thread.run_sync(lambda: server.release_all(http2=True))
        for response in [*held, *extra]:
            await drain(response)


@pytest.mark.parametrize("transport_type", [HTTPTransport, SyncHTTPTransport])
def test_max_connections_per_address_must_be_positive(transport_type: type) -> None:
    with pytest.raises(
        ValueError, match="max_connections_per_address must be positive"
    ):
        transport_type(max_connections_per_address=0)


@pytest.mark.anyio
@pytest.mark.parametrize("how", ["read", "aclose", "drop"])
async def test_stream_slot_released(server: PoolTestServer, how: str) -> None:
    """With one connection allowed and both of its slots held, a slot must come
    back when a response is read to the end, closed, or dropped unread, or the
    next request could never start."""
    connections = server.connections()
    streams = server.streams()
    async with HTTPTransport(
        http_version=HTTPVersion.HTTP2, max_connections_per_address=1
    ) as transport:
        held = [await open_stream(transport, server.url) for _ in range(STREAM_LIMIT)]
        await wait(server.streams, streams + STREAM_LIMIT, "streams")

        # The server finishes the oldest stream first.
        response = held.pop(0)
        if how == "read":
            server.release(1)
            await drain(response)
        elif how == "aclose":
            await response.aclose()
        else:
            del response
            gc.collect()

        with anyio.fail_after(5):
            held.append(await open_stream(transport, server.url))
        await wait(server.streams, streams + STREAM_LIMIT + 1, "streams")
        assert server.connections() == connections + 1

        server.release_all()
        for response in held:
            await drain(response)


def _overrides(*addresses: str) -> dict[str, list[str]]:
    return {HOST: list(addresses)}


def _connections(server: PoolTestServer, http_version: HTTPVersion) -> int:
    if http_version == HTTPVersion.HTTP1:
        return server.http1_connections()
    return server.connections()


@pytest.mark.anyio
@pytest.mark.parametrize("http_version", [HTTPVersion.HTTP1, HTTPVersion.HTTP2])
async def test_dns_load_balancing_connects_to_each_server(
    server: PoolTestServer, other_server: PoolTestServer, http_version: HTTPVersion
) -> None:
    before = (
        _connections(server, http_version),
        _connections(other_server, http_version),
    )
    async with HTTPTransport(
        http_version=http_version,
        enable_dns_load_balancing=True,
        dns_overrides=_overrides(server.address, other_server.address),
    ) as transport:
        for _ in range(6):
            await get(transport, f"http://{HOST}/short")
    assert (
        _connections(server, http_version),
        _connections(other_server, http_version),
    ) == (before[0] + 1, before[1] + 1)


@pytest.mark.anyio
@pytest.mark.parametrize("http_version", [HTTPVersion.HTTP1, HTTPVersion.HTTP2])
async def test_without_dns_load_balancing_one_server_is_used(
    server: PoolTestServer, other_server: PoolTestServer, http_version: HTTPVersion
) -> None:
    before = _connections(server, http_version) + _connections(
        other_server, http_version
    )
    async with HTTPTransport(
        http_version=http_version,
        dns_overrides=_overrides(server.address, other_server.address),
    ) as transport:
        for _ in range(6):
            await get(transport, f"http://{HOST}/short")
    assert (
        _connections(server, http_version) + _connections(other_server, http_version)
        == before + 1
    )


@pytest.mark.anyio
async def test_dns_load_balancing_with_connection_cap(
    server: PoolTestServer, other_server: PoolTestServer
) -> None:
    """A cap of one connection per address still allows one per server, and
    streams past both servers' limits wait rather than open more."""
    before = (server.connections(), other_server.connections())
    streams = server.streams() + other_server.streams()
    async with HTTPTransport(
        http_version=HTTPVersion.HTTP2,
        enable_dns_load_balancing=True,
        max_connections_per_address=1,
        dns_overrides=_overrides(server.address, other_server.address),
    ) as transport:
        url = f"http://{HOST}"
        held = [await open_stream(transport, url) for _ in range(STREAM_LIMIT)]
        extra: list[Response] = []

        async def open_extra() -> None:
            extra.append(await open_stream(transport, url))

        def total_streams() -> int:
            return server.streams() + other_server.streams()

        async with anyio.create_task_group() as tg:
            for _ in range(3):
                tg.start_soon(open_extra)
            await wait(total_streams, streams + 2 * STREAM_LIMIT, "streams")
            assert await to_thread.run_sync(settle, total_streams) == streams + 4
            assert (server.connections(), other_server.connections()) == (
                before[0] + 1,
                before[1] + 1,
            )
            server.release_all()
            other_server.release_all()
            for response in held:
                await drain(response)
        server.release_all()
        other_server.release_all()
        for response in extra:
            await drain(response)


@pytest.mark.anyio
async def test_dns_load_balancing_skips_unreachable_address(
    server: PoolTestServer,
) -> None:
    """The name also points at a port where nothing listens. Requests keep
    flowing on the live server without a connection attempt each time."""
    connections = server.connections()
    async with HTTPTransport(
        http_version=HTTPVersion.HTTP2,
        enable_dns_load_balancing=True,
        dns_overrides=_overrides(server.address, f"127.0.0.1:{free_port()}"),
    ) as transport:
        with anyio.fail_after(10):
            for _ in range(8):
                await get(transport, f"http://{HOST}/short")
    # One connection, plus at most one more from the attempt at the dead
    # address that fell back to the live one before it was marked unreachable.
    assert server.connections() - connections <= 2


def test_dns_overrides_invalid_address() -> None:
    with pytest.raises(ValueError, match="not an address"):
        HTTPTransport(dns_overrides={HOST: ["not an address"]})
