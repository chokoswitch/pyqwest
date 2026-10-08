"""A pyvoy server configured for connection pool tests."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import socket
import time
import urllib.request
from typing import TYPE_CHECKING

from pyvoy import PyvoyServer

from pyqwest import HTTPVersion, SyncHTTPTransport, SyncRequest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

WAIT_TIMEOUT = 5.0


class PoolTestServer(PyvoyServer):
    def __init__(self, *, max_concurrent_streams: int) -> None:
        self.max_concurrent_streams = max_concurrent_streams
        super().__init__("tests.apps.asgi.streams", lifespan=False)

    def get_envoy_config(self) -> dict:
        config = super().get_envoy_config()
        listener = config["static_resources"]["listeners"][0]
        http = listener["filter_chains"][0]["filters"][0]["typed_config"]
        http["http2_protocol_options"] = {
            "max_concurrent_streams": self.max_concurrent_streams
        }
        return config

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.listener_port}"

    @property
    def url(self) -> str:
        return f"http://{self.address}"

    def _admin(self, path: str) -> bytes:
        with urllib.request.urlopen(f"http://{self._admin_address}{path}") as response:
            return response.read()

    def _stat(self, name: str) -> int:
        pattern = f"^{re.escape(name)}$"
        data = json.loads(self._admin(f"/stats?format=json&filter={pattern}"))
        return next(stat["value"] for stat in data["stats"] if stat["name"] == name)

    def connections(self) -> int:
        """HTTP/2 connections accepted so far."""
        return self._stat("http.ingress_http.downstream_cx_http2_total")

    def streams(self) -> int:
        """HTTP/2 request streams received so far."""
        return self._stat("http.ingress_http.downstream_rq_http2_total")

    def http1_connections(self) -> int:
        """HTTP/1 connections accepted so far, including those made by
        `release` unless it is told to use HTTP/2."""
        return self._stat("http.ingress_http.downstream_cx_http1_total")

    def http1_requests(self) -> int:
        return self._stat("http.ingress_http.downstream_rq_http1_total")

    def release(self, n: int = 1, *, http2: bool = False) -> None:
        """Finishes the `n` oldest open `/stream` responses. Uses HTTP/1 unless
        `http2`, so the request never counts toward the version under test."""
        self._side_request(f"/release?n={n}", http2=http2)

    def release_all(self, *, http2: bool = False) -> None:
        self._side_request("/release?all=1", http2=http2)

    def _side_request(self, path: str, *, http2: bool) -> None:
        if http2:
            with SyncHTTPTransport(http_version=HTTPVersion.HTTP2) as transport:
                response = transport.execute_sync(
                    SyncRequest("GET", f"{self.url}{path}")
                )
                for _ in response.content:
                    pass
            return
        with urllib.request.urlopen(f"{self.url}{path}") as response:  # noqa: S310
            response.read()


@contextlib.contextmanager
def run_pool_server(*, max_concurrent_streams: int) -> Iterator[PoolTestServer]:
    """Runs a `PoolTestServer` for the duration of the block."""
    server = PoolTestServer(max_concurrent_streams=max_concurrent_streams)
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


def free_port() -> int:
    """A port nothing listens on right now."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for(getter: Callable[[], int], expected: int, *, what: str) -> None:
    """Waits until an Envoy counter reaches `expected`; they update a moment
    after the wire."""
    deadline = time.monotonic() + WAIT_TIMEOUT
    while True:
        value = getter()
        if value == expected:
            return
        if time.monotonic() > deadline:
            msg = f"{what} is {value}, expected {expected}"
            raise TimeoutError(msg)
        time.sleep(0.01)


def settle(getter: Callable[[], int], *, seconds: float = 0.5) -> int:
    """Returns the counter's value after it has had `seconds` to change."""
    time.sleep(seconds)
    return getter()
