"""Background task processing with Celery."""

from typing import Optional, Dict, Any
from celery import Celery
from celery.signals import task_failure, task_success

from rag_chatbot.core.config import get_settings
from rag_chatbot.core.logging_config import get_logger

logger = get_logger(__name__)

# Initialize Celery
settings = get_settings()
redis_url = settings.redis_url or "redis://localhost:6379/0"

celery_app = Celery(
    "rag_chatbot",
    broker=redis_url,
    backend=redis_url,
    include=["rag_chatbot.tasks.ingestion"],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)


@task_success.connect
def handle_task_success(sender=None, result=None, **kwargs):
    """Handle successful task completion."""
    logger.info(f"Task {sender.request.id} completed successfully")


@task_failure.connect
def handle_task_failure(sender=None, task_id=None, exception=None, **kwargs):
    """Handle task failure."""
    logger.error(f"Task {task_id} failed: {exception}")


class TaskManager:
    """Manage background tasks."""

    def __init__(self):
        """Initialize task manager."""
        self.celery = celery_app

    def submit_ingestion_task(
        self,
        file_path: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Submit document ingestion task.

        Args:
            file_path: Path to file to ingest
            user_id: User submitting the task
            metadata: Additional metadata

        Returns:
            Task ID
        """
        from rag_chatbot.tasks.ingestion import ingest_document_task

        task = ingest_document_task.delay(file_path, user_id, metadata)
        logger.info(f"Submitted ingestion task: {task.id}")
        return task.id

    def submit_batch_ingestion(
        self,
        file_paths: list,
        user_id: str,
    ) -> list:
        """Submit multiple ingestion tasks.

        Args:
            file_paths: List of file paths
            user_id: User submitting

        Returns:
            List of task IDs
        """
        from rag_chatbot.tasks.ingestion import ingest_document_task

        task_ids = []
        for file_path in file_paths:
            task = ingest_document_task.delay(file_path, user_id)
            task_ids.append(task.id)

        logger.info(f"Submitted {len(task_ids)} ingestion tasks")
        return task_ids

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status.

        Args:
            task_id: Task ID

        Returns:
            Task status dictionary
        """
        result = self.celery.AsyncResult(task_id)

        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
            "traceback": result.traceback if result.failed() else None,
        }

    def revoke_task(self, task_id: str, terminate: bool = False) -> bool:
        """Revoke a task.

        Args:
            task_id: Task ID to revoke
            terminate: Whether to terminate running task

        Returns:
            True if revoked
        """
        self.celery.control.revoke(task_id, terminate=terminate)
        logger.info(f"Revoked task: {task_id}")
        return True


# Global task manager
task_manager = TaskManager()
