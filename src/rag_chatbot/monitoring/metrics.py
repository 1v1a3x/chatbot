"""Production monitoring and observability."""

import time
from typing import Callable, Any
from functools import wraps
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger

logger = get_logger(__name__)


# Prometheus metrics
REQUEST_COUNT = Counter(
    "rag_chatbot_requests_total",
    "Total requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "rag_chatbot_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

SEARCH_DURATION = Histogram(
    "rag_chatbot_search_duration_seconds",
    "Search duration",
    ["search_type"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

LLM_DURATION = Histogram(
    "rag_chatbot_llm_duration_seconds",
    "LLM request duration",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

LLM_TOKENS = Counter(
    "rag_chatbot_llm_tokens_total",
    "Total LLM tokens used",
    ["type"],  # prompt, completion
)

ACTIVE_CONNECTIONS = Gauge(
    "rag_chatbot_active_connections",
    "Number of active connections",
)

CACHE_HITS = Counter(
    "rag_chatbot_cache_hits_total",
    "Cache hits",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "rag_chatbot_cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

DOCUMENTS_INGESTED = Counter(
    "rag_chatbot_documents_ingested_total",
    "Documents ingested",
    ["file_type"],
)

ERRORS = Counter(
    "rag_chatbot_errors_total",
    "Total errors",
    ["error_type"],
)

APP_INFO = Info("rag_chatbot_app", "Application information")


def init_metrics(app_version: str):
    """Initialize application metrics."""
    APP_INFO.info({"version": app_version})
    logger.info("Metrics initialized")


@contextmanager
def measure_duration(histogram: Histogram, labels: tuple = ()):
    """Context manager to measure operation duration.

    Args:
        histogram: Histogram to observe
        labels: Labels for the histogram
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if labels:
            histogram.labels(*labels).observe(duration)
        else:
            histogram.observe(duration)


def timed(metric: Histogram, labels: tuple = ()):
    """Decorator to time function execution.

    Args:
        metric: Histogram metric
        labels: Labels for the metric
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with measure_duration(metric, labels):
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with measure_duration(metric, labels):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def track_request(method: str, endpoint: str, status: int, duration: float):
    """Track HTTP request metrics.

    Args:
        method: HTTP method
        endpoint: Endpoint path
        status: Response status code
        duration: Request duration in seconds
    """
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)


def track_search(search_type: str, duration: float):
    """Track search operation.

    Args:
        search_type: Type of search (keyword, semantic, hybrid)
        duration: Search duration in seconds
    """
    SEARCH_DURATION.labels(search_type=search_type).observe(duration)


def track_llm_request(duration: float, prompt_tokens: int = 0, completion_tokens: int = 0):
    """Track LLM request.

    Args:
        duration: Request duration
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
    """
    LLM_DURATION.observe(duration)
    if prompt_tokens:
        LLM_TOKENS.labels(type="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(type="completion").inc(completion_tokens)


def track_cache_hit(cache_type: str):
    """Track cache hit."""
    CACHE_HITS.labels(cache_type=cache_type).inc()


def track_cache_miss(cache_type: str):
    """Track cache miss."""
    CACHE_MISSES.labels(cache_type=cache_type).inc()


def track_error(error_type: str):
    """Track error."""
    ERRORS.labels(error_type=error_type).inc()


def get_metrics():
    """Get metrics in Prometheus format."""
    return generate_latest()


# Sentry integration
class SentryManager:
    """Manage Sentry error tracking."""

    def __init__(self, dsn: str = None):
        """Initialize Sentry.

        Args:
            dsn: Sentry DSN
        """
        self.dsn = dsn
        self._initialized = False

    def init(self):
        """Initialize Sentry SDK."""
        if not self.dsn or self._initialized:
            return

        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=logging.INFO,
                event_level=logging.ERROR,
            )

            sentry_sdk.init(
                dsn=self.dsn,
                integrations=[sentry_logging],
                traces_sample_rate=1.0,
                profiles_sample_rate=0.1,
            )

            self._initialized = True
            logger.info("Sentry initialized")

        except ImportError:
            logger.warning("sentry-sdk not installed")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")

    def capture_exception(self, exception: Exception, context: dict = None):
        """Capture exception to Sentry.

        Args:
            exception: Exception to capture
            context: Additional context
        """
        if not self._initialized:
            return

        try:
            import sentry_sdk

            with sentry_sdk.push_scope() as scope:
                if context:
                    for key, value in context.items():
                        scope.set_extra(key, value)
                sentry_sdk.capture_exception(exception)
        except Exception as e:
            logger.error(f"Failed to capture exception: {e}")


# Initialize Sentry if configured
settings = get_settings()
sentry = SentryManager(getattr(settings, "sentry_dsn", None))


import asyncio
import logging
