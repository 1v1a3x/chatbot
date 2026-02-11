"""Health checks and monitoring endpoints."""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import asyncio

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: str  # healthy, degraded, unhealthy
    response_time_ms: float
    details: Dict[str, Any]
    error: str = None


class HealthChecker:
    """Perform health checks on services."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: List[callable] = []
        self._last_check: Dict[str, HealthCheck] = {}

    def register(self, name: str, check_func: callable):
        """Register a health check.

        Args:
            name: Check name
            check_func: Async function returning (status, details)
        """
        self.checks.append((name, check_func))
        logger.info(f"Registered health check: {name}")

    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks.

        Returns:
            Health status report
        """
        start_time = datetime.utcnow()
        results = []

        # Run checks concurrently
        tasks = [self._run_check(name, func) for name, func in self.checks]

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in check_results:
            if isinstance(result, Exception):
                logger.error(f"Health check failed: {result}")
                continue
            results.append(result)
            self._last_check[result.name] = result

        # Determine overall status
        statuses = [r.status for r in results]
        if all(s == "healthy" for s in statuses):
            overall = "healthy"
        elif any(s == "unhealthy" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        return {
            "status": overall,
            "timestamp": start_time.isoformat(),
            "checks": {
                r.name: {
                    "status": r.status,
                    "response_time_ms": r.response_time_ms,
                    "details": r.details,
                    "error": r.error,
                }
                for r in results
            },
        }

    async def _run_check(self, name: str, check_func: callable) -> HealthCheck:
        """Run a single health check."""
        import time

        start = time.time()

        try:
            status, details = await check_func()
            error = None
        except Exception as e:
            status = "unhealthy"
            details = {}
            error = str(e)
            logger.error(f"Health check {name} failed: {e}")

        elapsed_ms = (time.time() - start) * 1000

        return HealthCheck(
            name=name,
            status=status,
            response_time_ms=elapsed_ms,
            details=details,
            error=error,
        )


# Global health checker
health_checker = HealthChecker()


async def check_elasticsearch() -> tuple:
    """Check Elasticsearch health."""
    from elasticsearch import AsyncElasticsearch
    from rag_chatbot.core.config import get_settings

    settings = get_settings()

    es = AsyncElasticsearch(
        settings.es_url,
        api_key=settings.es_api_key,
        verify_certs=settings.es_verify_ssl,
    )

    try:
        health = await es.cluster.health()
        status = "healthy" if health["status"] in ["green", "yellow"] else "degraded"
        details = {
            "cluster_status": health["status"],
            "nodes": health.get("number_of_nodes", 0),
        }
        return status, details
    except Exception as e:
        return "unhealthy", {"error": str(e)}
    finally:
        await es.close()


async def check_openai() -> tuple:
    """Check OpenAI API health."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {get_settings().openai_api_key}"},
                timeout=10,
            )

            if response.status_code == 200:
                return "healthy", {"api": "accessible"}
            else:
                return "degraded", {"status_code": response.status_code}
    except Exception as e:
        return "unhealthy", {"error": str(e)}


async def check_redis() -> tuple:
    """Check Redis health."""
    settings = get_settings()

    if not settings.redis_url:
        return "healthy", {"message": "Redis not configured"}

    try:
        import redis

        r = redis.from_url(settings.redis_url)
        r.ping()
        info = r.info()
        return "healthy", {
            "version": info.get("redis_version"),
            "connected_clients": info.get("connected_clients"),
        }
    except ImportError:
        return "healthy", {"message": "redis not installed"}
    except Exception as e:
        return "unhealthy", {"error": str(e)}


# Register default checks
health_checker.register("elasticsearch", check_elasticsearch)
health_checker.register("openai", check_openai)
health_checker.register("redis", check_redis)
