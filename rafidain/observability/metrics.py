"""Prometheus metrics registry (singleton-safe across tests & reloads)."""

from __future__ import annotations

from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_REGISTRY = CollectorRegistry()
_METRICS: dict[str, Any] | None = None


def metrics() -> dict[str, Any]:
    """Lazily create the metric set exactly once per process."""
    global _METRICS
    if _METRICS is None:
        _METRICS = {
            "http_requests": Counter(
                "rafidain_http_requests_total",
                "HTTP requests by method/path/status",
                labelnames=("method", "path", "status"),
                registry=_REGISTRY,
            ),
            "http_latency": Histogram(
                "rafidain_http_request_seconds",
                "HTTP request latency",
                labelnames=("method", "path"),
                registry=_REGISTRY,
            ),
            "ws_sessions_active": Gauge(
                "rafidain_ws_sessions_active",
                "Active streaming sessions",
                registry=_REGISTRY,
            ),
            "blocks_processed": Counter(
                "rafidain_stream_blocks_total",
                "Audio blocks fed to the engine",
                labelnames=("profile",),
                registry=_REGISTRY,
            ),
            "block_latency": Histogram(
                "rafidain_block_processing_seconds",
                "Per-block engine processing time",
                labelnames=("profile",),
                buckets=(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
                registry=_REGISTRY,
            ),
            "block_rtf": Gauge(
                "rafidain_block_rtf",
                "Real-time factor per session (EWMA)",
                labelnames=("session_id",),
                registry=_REGISTRY,
            ),
            "overflows": Counter(
                "rafidain_input_overflows_total",
                "Backpressure overflow events",
                registry=_REGISTRY,
            ),
        }
    return _METRICS


def render_metrics() -> bytes:
    return generate_latest(_REGISTRY)
