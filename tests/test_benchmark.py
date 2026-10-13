from __future__ import annotations

import asyncio
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import aiohttp
import httpx
import pytest
from anyio import to_thread

from pyqwest import Client, HTTPTransport, HTTPVersion, SyncClient, SyncHTTPTransport
from pyqwest.httpx import AsyncPyqwestTransport, PyqwestTransport

try:
    import uvloop

    new_event_loop = uvloop.new_event_loop
except ImportError:
    import winloop  # pyright: ignore[reportMissingImports]

    new_event_loop = winloop.new_event_loop

# Other reqwest wrappers, which are only installed on the Python versions they
# publish wheels for.
try:
    import pyreqwest.client
    import ry
    import wreq
    import wreq.blocking
    import wreq.tls
except ImportError:
    HAS_REQWEST_WRAPPERS = False
else:
    HAS_REQWEST_WRAPPERS = True

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

    import pytest_benchmark.fixture

    from .conftest import Certs

pytestmark = [
    # Benchmarks drive their own asyncio.Runner.
    pytest.mark.asyncio_only,
    pytest.mark.parametrize("http_scheme", ["http"], indirect=True),
    pytest.mark.parametrize("http_version", ["h1", "h2", "h3"], indirect=True),
]

CONCURRENCY = 10
TASK_SIZE = 30

REQWEST_WRAPPERS = ("pyreqwest", "ry", "wreq")

if TYPE_CHECKING:
    AsyncBenchmarkClient = (
        Client
        | httpx.AsyncClient
        | aiohttp.ClientSession
        | pyreqwest.client.Client
        | ry.Client
        | wreq.Client
    )
    SyncBenchmarkClient = (
        SyncClient
        | httpx.Client
        | pyreqwest.client.SyncClient
        | ry.BlockingClient
        | wreq.blocking.Client
    )


@pytest.fixture(
    params=["pyqwest", "aiohttp", "httpx", "httpx_pyqwest", *REQWEST_WRAPPERS],
    scope="module",
)
def library(request: pytest.FixtureRequest) -> str:
    if request.param in REQWEST_WRAPPERS and not HAS_REQWEST_WRAPPERS:
        pytest.skip(f"{request.param} is not installed")
    return request.param


@pytest.fixture(scope="module")
def async_runner() -> Iterator[asyncio.Runner]:
    with asyncio.Runner(loop_factory=new_event_loop) as runner:
        yield runner


@pytest.fixture(scope="module")
def sync_runner() -> Iterator[ThreadPoolExecutor]:
    with ThreadPoolExecutor(CONCURRENCY) as executor:
        yield executor


@pytest.fixture(scope="module")
async def benchmark_client_async(
    anyio_backend: object,  # noqa: ARG001  # runs this fixture on the test's backend
    async_client: Client,
    async_transport: HTTPTransport,
    certs: Certs,
    http_version: HTTPVersion | None,
    library: str,
    async_runner: asyncio.Runner,
) -> AsyncIterator[AsyncBenchmarkClient]:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations(cadata=certs.ca.decode())
    match library:
        case "aiohttp":
            if http_version != HTTPVersion.HTTP1:
                pytest.skip("aiohttp only supports HTTP/1")

            async def _create_session() -> aiohttp.ClientSession:
                return aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(ssl=ssl_ctx), timeout=None
                )

            session = await to_thread.run_sync(async_runner.run, _create_session())
            try:
                yield session
            finally:
                await to_thread.run_sync(async_runner.run, session.close())
        case "httpx":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("httpx does not support HTTP/3")
            async with httpx.AsyncClient(
                verify=ssl_ctx,
                http1=(http_version in (HTTPVersion.HTTP1, None)),
                http2=(http_version in (HTTPVersion.HTTP2, None)),
                timeout=None,  # noqa: S113
                trust_env=False,
            ) as client:
                yield client
        case "httpx_pyqwest":
            async with httpx.AsyncClient(
                transport=AsyncPyqwestTransport(async_transport)
            ) as client:
                yield client
        case "pyreqwest":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("pyreqwest does not support HTTP/3")
            builder = pyreqwest.client.ClientBuilder().add_root_certificate_pem(
                certs.ca
            )
            if http_version == HTTPVersion.HTTP1:
                builder = builder.http1_only()
            elif http_version == HTTPVersion.HTTP2:
                builder = builder.http2_prior_knowledge()

            async def _create_client() -> pyreqwest.client.Client:
                return builder.build()

            client = await to_thread.run_sync(async_runner.run, _create_client())
            try:
                yield client
            finally:
                client.close()
        case "ry":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("ry does not support HTTP/3")
            yield ry.Client(
                http1_only=(http_version == HTTPVersion.HTTP1),
                http2_prior_knowledge=(http_version == HTTPVersion.HTTP2),
                tls_certs_only=[ry.Certificate.from_pem(certs.ca)],
            )
        case "wreq":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("wreq does not support HTTP/3")
            yield wreq.Client(
                http1_only=(http_version == HTTPVersion.HTTP1),
                http2_only=(http_version == HTTPVersion.HTTP2),
                tls_verify=wreq.tls.CertStore(pem_certs=[certs.ca.decode()]),
            )
        case "pyqwest":
            yield async_client


@pytest.fixture(scope="module")
def benchmark_client_sync(
    sync_client: SyncClient,
    sync_transport: SyncHTTPTransport,
    certs: Certs,
    http_version: HTTPVersion | None,
    library: str,
) -> Iterator[SyncBenchmarkClient]:
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.load_verify_locations(cadata=certs.ca.decode())
    match library:
        case "aiohttp":
            pytest.skip("aiohttp does not have a sync client")
        case "httpx":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("httpx does not support HTTP/3")
            with httpx.Client(
                verify=ssl_ctx,
                http1=(http_version in (HTTPVersion.HTTP1, None)),
                http2=(http_version in (HTTPVersion.HTTP2, None)),
                timeout=None,  # noqa: S113
                trust_env=False,
            ) as client:
                yield client
        case "httpx_pyqwest":
            with httpx.Client(transport=PyqwestTransport(sync_transport)) as client:
                yield client
        case "pyreqwest":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("pyreqwest does not support HTTP/3")
            builder = pyreqwest.client.SyncClientBuilder().add_root_certificate_pem(
                certs.ca
            )
            if http_version == HTTPVersion.HTTP1:
                builder = builder.http1_only()
            elif http_version == HTTPVersion.HTTP2:
                builder = builder.http2_prior_knowledge()
            with builder.build() as client:
                yield client
        case "ry":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("ry does not support HTTP/3")
            yield ry.BlockingClient(
                http1_only=(http_version == HTTPVersion.HTTP1),
                http2_prior_knowledge=(http_version == HTTPVersion.HTTP2),
                tls_certs_only=[ry.Certificate.from_pem(certs.ca)],
            )
        case "wreq":
            if http_version == HTTPVersion.HTTP3:
                pytest.skip("wreq does not support HTTP/3")
            yield wreq.blocking.Client(
                http1_only=(http_version == HTTPVersion.HTTP1),
                http2_only=(http_version == HTTPVersion.HTTP2),
                tls_verify=wreq.tls.CertStore(pem_certs=[certs.ca.decode()]),
            )
        case "pyqwest":
            yield sync_client


METHOD = "POST"
HEADERS = [("content-type", "text/plain"), ("x-hello", "rust"), ("x-hello", "python")]
# ry and wreq take a mapping rather than a header list.
HEADERS_RY = {"content-type": "text/plain", "x-hello": ["rust", "python"]}
HEADERS_WREQ = {"content-type": "text/plain", "x-hello": "rust, python"}


async def _generate_body_async(size: int) -> AsyncIterator[bytes]:
    chunk = b"A" * 1024
    for _ in range(size // 1024):
        yield chunk


def _generate_body_sync(size: int) -> Iterator[bytes]:
    chunk = b"A" * 1024
    for _ in range(size // 1024):
        yield chunk


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="asyncio.Runner requires Python 3.11+"
)
@pytest.mark.parametrize("content_size", [0, 1024, 1024 * 1024])
def test_benchmark_async(
    benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    benchmark_client_async: AsyncBenchmarkClient,
    url: str,
    content_size: int,
    async_runner: asyncio.Runner,
) -> None:
    target_url = f"{url}/echo"

    # A generator is consumed by one request, so each request gets its own.
    def make_body() -> bytes | AsyncIterator[bytes]:
        match content_size:
            case 0:
                return b""
            case 1024:
                return b"A" * 1024
            case _:
                return _generate_body_async(content_size)

    execute_request: Callable[[], Awaitable[None]]
    match benchmark_client_async:
        case Client():

            async def execute_request_pyqwest() -> None:
                for _ in range(TASK_SIZE):
                    async with benchmark_client_async.stream(
                        METHOD, target_url, HEADERS, make_body()
                    ) as res:
                        assert res.status == 200
                        received = 0
                        async for chunk in res.content:
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_pyqwest
        case aiohttp.ClientSession():

            async def execute_request_aiohttp() -> None:
                for _ in range(TASK_SIZE):
                    async with benchmark_client_async.request(
                        METHOD, target_url, headers=HEADERS, data=make_body()
                    ) as res:
                        assert res.status == 200
                        received = 0
                        async for chunk in res.content.iter_chunked(1024):
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_aiohttp
        case httpx.AsyncClient():

            async def execute_request_httpx() -> None:
                for _ in range(TASK_SIZE):
                    async with benchmark_client_async.stream(
                        METHOD, target_url, headers=HEADERS, content=make_body()
                    ) as res:
                        assert res.status_code == 200
                        received = 0
                        async for chunk in res.aiter_bytes():
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_httpx
        case pyreqwest.client.Client():

            async def execute_request_pyreqwest() -> None:
                for _ in range(TASK_SIZE):
                    request = benchmark_client_async.request(
                        METHOD, target_url
                    ).headers(HEADERS)
                    body = make_body()
                    if isinstance(body, bytes):
                        request = request.body_bytes(body)
                    else:
                        request = request.body_stream(body)
                    async with request.build_streamed() as res:
                        assert res.status == 200
                        received = 0
                        while (chunk := await res.body_reader.read_chunk()) is not None:
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_pyreqwest
        case ry.Client():

            async def execute_request_ry() -> None:
                for _ in range(TASK_SIZE):
                    res = await benchmark_client_async.fetch(
                        target_url, method=METHOD, headers=HEADERS_RY, body=make_body()
                    )
                    assert res.status == 200
                    received = 0
                    async for chunk in res.bytes_stream():
                        received += len(chunk)
                    assert received == content_size

            execute_request = execute_request_ry
        case wreq.Client():

            async def execute_request_wreq() -> None:
                for _ in range(TASK_SIZE):
                    async with await benchmark_client_async.request(
                        wreq.Method.POST,
                        target_url,
                        headers=HEADERS_WREQ,
                        body=make_body(),
                    ) as res:
                        assert res.status.as_int() == 200
                        received = 0
                        async with res.stream() as stream:
                            async for chunk in stream:
                                received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_wreq

    async def execute_requests() -> None:
        tasks = [asyncio.create_task(execute_request()) for _ in range(CONCURRENCY)]
        await asyncio.gather(*tasks)

    @benchmark
    def run_benchmark() -> None:
        async_runner.run(execute_requests())


@pytest.mark.parametrize("content_size", [0, 1024, 1024 * 1024])
def test_benchmark_sync(
    benchmark: pytest_benchmark.fixture.BenchmarkFixture,
    benchmark_client_sync: SyncBenchmarkClient,
    url: str,
    content_size: int,
    sync_runner: ThreadPoolExecutor,
) -> None:
    target_url = f"{url}/echo"

    # A generator is consumed by one request, so each request gets its own.
    def make_body() -> bytes | Iterator[bytes]:
        match content_size:
            case 0:
                return b""
            case 1024:
                return b"A" * 1024
            case _:
                return _generate_body_sync(content_size)

    execute_request: Callable[[], None]
    match benchmark_client_sync:
        case SyncClient():

            def execute_request_pyqwest() -> None:
                for _ in range(TASK_SIZE):
                    with benchmark_client_sync.stream(
                        METHOD, target_url, HEADERS, make_body()
                    ) as res:
                        assert res.status == 200
                        received = 0
                        for chunk in res.content:
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_pyqwest
        case httpx.Client():

            def execute_request_httpx() -> None:
                for _ in range(TASK_SIZE):
                    with benchmark_client_sync.stream(
                        METHOD, target_url, headers=HEADERS, content=make_body()
                    ) as res:
                        assert res.status_code == 200
                        received = 0
                        for chunk in res.iter_bytes():
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_httpx
        case pyreqwest.client.SyncClient():

            def execute_request_pyreqwest() -> None:
                for _ in range(TASK_SIZE):
                    request = benchmark_client_sync.request(METHOD, target_url).headers(
                        HEADERS
                    )
                    body = make_body()
                    if isinstance(body, bytes):
                        request = request.body_bytes(body)
                    else:
                        request = request.body_stream(body)
                    with request.build_streamed() as res:
                        assert res.status == 200
                        received = 0
                        while (chunk := res.body_reader.read_chunk()) is not None:
                            received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_pyreqwest
        case ry.BlockingClient():

            def execute_request_ry() -> None:
                for _ in range(TASK_SIZE):
                    res = benchmark_client_sync.fetch(
                        target_url, method=METHOD, headers=HEADERS_RY, body=make_body()
                    )
                    assert res.status == 200
                    received = 0
                    for chunk in res.bytes_stream():
                        received += len(chunk)
                    assert received == content_size

            execute_request = execute_request_ry
        case wreq.blocking.Client():

            def execute_request_wreq() -> None:
                for _ in range(TASK_SIZE):
                    with benchmark_client_sync.request(
                        wreq.Method.POST,
                        target_url,
                        headers=HEADERS_WREQ,
                        body=make_body(),
                    ) as res:
                        assert res.status.as_int() == 200
                        received = 0
                        with res.stream() as stream:
                            for chunk in stream:
                                received += len(chunk)
                        assert received == content_size

            execute_request = execute_request_wreq

    @benchmark
    def execute_requests() -> None:
        tasks = [sync_runner.submit(execute_request) for _ in range(CONCURRENCY)]
        for task in tasks:
            task.result()
