"""Helpers for reading the metrics pyqwest records with OpenTelemetry. They
validate what they read in the same way tests do, so they use assert."""

# ruff: noqa: S101

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import anyio

if TYPE_CHECKING:
    from collections.abc import Callable

    from opentelemetry.sdk.metrics._internal.point import Metric, Sum
    from opentelemetry.test.test_base import TestBase


def get_http_metrics(otel_test_base: TestBase) -> list[Metric]:
    metrics = cast("list[Metric]", otel_test_base.get_sorted_metrics())
    return [metric for metric in metrics if metric.name.startswith("http.client.")]


def get_http_metric(metrics: list[Metric], name: str) -> Metric:
    """Finds a metric by name in an already collected list. Collecting again
    would discard exemplars."""
    found = [m for m in metrics if m.name == name]
    assert len(found) == 1, f"expected one {name} metric, got {metrics}"
    return found[0]


def open_connections(otel_test_base: TestBase, base_attrs: dict) -> dict[str, int]:
    """Returns the `http.client.open_connections` values keyed by connection
    state for the data points whose other attributes are `base_attrs`. The
    metric is process-wide, so other transports' connections are ignored.
    Returns an empty dict when nothing matched."""
    metrics = [
        m
        for m in get_http_metrics(otel_test_base)
        if m.name == "http.client.open_connections"
    ]
    if not metrics:
        return {}
    (metric,) = metrics
    assert metric.unit == "{connection}"
    assert metric.description == (
        "Number of outbound HTTP connections that are currently active or idle "
        "on the client."
    )
    result: dict[str, int] = {}
    for point in cast("Sum", metric.data).data_points:
        attrs = dict(point.attributes or {})
        state = attrs.pop("http.connection.state")
        if attrs == base_attrs:
            result[cast("str", state)] = cast("int", point.value)
    return result


async def wait_until(getter: Callable[[], object], expected: object, what: str) -> None:
    """Waits until `getter` returns `expected`. The pool returns connections on
    the tokio runtime, which happens a moment after the request completes."""
    deadline = time.monotonic() + 5
    while (actual := getter()) != expected:
        if time.monotonic() > deadline:
            msg = f"timed out waiting for {what}: got {actual!r}, expected {expected!r}"
            raise AssertionError(msg)
        await anyio.sleep(0.01)
