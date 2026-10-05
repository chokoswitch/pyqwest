from __future__ import annotations

from typing import TYPE_CHECKING

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

from ._otel import open_connections, wait_until
from ._pool_server import PoolTestServer, run_pool_server

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.test.test_base import TestBase

pytestmark = [
    pytest.mark.order(1),
    pytest.mark.parametrize(
        "http_version", [HTTPVersion.HTTP1, HTTPVersion.HTTP2], ids=["h1", "h2"]
    ),
    pytest.mark.parametrize("client_type", ["async", "sync"]),
]


@pytest.fixture(scope="module")
def server() -> Iterator[PoolTestServer]:
    with run_pool_server(max_concurrent_streams=100) as server:
        yield server


@pytest.fixture(scope="module")
def other_server() -> Iterator[PoolTestServer]:
    with run_pool_server(max_concurrent_streams=100) as server:
        yield server


def drain_sync(response: SyncResponse) -> None:
    for _ in response.content:
        pass


async def drain(response: Response) -> None:
    async for _ in response.content:
        pass


@pytest.mark.anyio
async def test_connections(
    server: PoolTestServer,
    otel_test_base: TestBase,
    client_type: str,
    http_version: HTTPVersion,
) -> None:
    base_attrs = {
        "server.address": "127.0.0.1",
        "server.port": server.listener_port,
        "network.protocol.version": "2" if http_version == HTTPVersion.HTTP2 else "1.1",
        "network.peer.address": "127.0.0.1",
    }

    short_url = f"{server.url}/short"
    stream_url = f"{server.url}/stream"

    def get_open() -> dict[str, int]:
        return open_connections(otel_test_base, base_attrs)

    transport: HTTPTransport | SyncHTTPTransport
    if client_type == "sync":
        transport = SyncHTTPTransport(http_version=http_version)
    else:
        transport = HTTPTransport(http_version=http_version)
    try:
        assert get_open() == {}

        # A completed request leaves its connection idle in the pool.
        if isinstance(transport, SyncHTTPTransport):
            sync_transport = transport

            def get_short() -> None:
                drain_sync(sync_transport.execute_sync(SyncRequest("GET", short_url)))

            await to_thread.run_sync(get_short)
        else:
            await drain(await transport.execute(Request("GET", short_url)))
        await wait_until(get_open, {"active": 0, "idle": 1}, "an idle connection")

        # A connection is busy while its response is still streaming, and idle
        # again once the response has finished and been read.
        if isinstance(transport, SyncHTTPTransport):
            sync_response = await to_thread.run_sync(
                transport.execute_sync, SyncRequest("GET", stream_url)
            )
            await wait_until(get_open, {"active": 1, "idle": 0}, "a busy connection")
            server.release_all()
            await to_thread.run_sync(drain_sync, sync_response)
            del sync_response
        else:
            response = await transport.execute(Request("GET", stream_url))
            await wait_until(get_open, {"active": 1, "idle": 0}, "a busy connection")
            server.release_all()
            await drain(response)
            del response
        await wait_until(get_open, {"active": 0, "idle": 1}, "an idle connection")
    finally:
        if isinstance(transport, SyncHTTPTransport):
            transport.close()
        else:
            await transport.aclose()

    # Closing the transport closes its connection, so nothing is reported as
    # open any more.
    assert get_open() == {}


@pytest.mark.anyio
async def test_connections_to_each_server(
    server: PoolTestServer,
    other_server: PoolTestServer,
    otel_test_base: TestBase,
    client_type: str,
    http_version: HTTPVersion,
) -> None:
    """DNS load balancing opens a connection to each server the host resolves
    to. Both test servers are on the same IP, so the metric reports them under
    one peer address, as two connections."""
    host = "pool.test"
    base_attrs = {
        "server.address": host,
        "server.port": 80,
        "network.protocol.version": "2" if http_version == HTTPVersion.HTTP2 else "1.1",
        "network.peer.address": "127.0.0.1",
    }
    url = f"http://{host}/short"
    dns_overrides = {host: [server.address, other_server.address]}

    def two_idle() -> dict[str, int]:
        return open_connections(otel_test_base, base_attrs)

    if client_type == "sync":
        with SyncHTTPTransport(
            http_version=http_version,
            enable_dns_load_balancing=True,
            dns_overrides=dns_overrides,
        ) as sync_transport:
            for _ in range(4):
                await to_thread.run_sync(
                    lambda: drain_sync(
                        sync_transport.execute_sync(SyncRequest("GET", url))
                    )
                )
            await wait_until(two_idle, {"active": 0, "idle": 2}, "two idle connections")
    else:
        async with HTTPTransport(
            http_version=http_version,
            enable_dns_load_balancing=True,
            dns_overrides=dns_overrides,
        ) as transport:
            for _ in range(4):
                await drain(await transport.execute(Request("GET", url)))
            await wait_until(two_idle, {"active": 0, "idle": 2}, "two idle connections")
