"""Structured logging with correlation IDs and security redaction."""

import logging
import logging.handlers
import re
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

# Context variable for correlation ID
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


class RedactingFilter(logging.Filter):
    """Filter that redacts sensitive information from log records."""
    
    SENSITIVE_PATTERNS = [
        (r'sk-[a-zA-Z0-9]{20,}', '***REDACTED_OPENAI_KEY***'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}', 'api_key="***REDACTED***"'),
        (r'[a-zA-Z0-9]{40,}:[a-zA-Z0-9]{40,}', '***REDACTED_ES_KEY***'),
        (r'Bearer\s+[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+', 'Bearer ***REDACTED_JWT***'),
        (r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+', 'password="***REDACTED***"'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive data from log message."""
        if isinstance(record.msg, str):
            for pattern, replacement in self.SENSITIVE_PATTERNS:
                record.msg = re.sub(pattern, replacement, record.msg, flags=re.IGNORECASE)
        
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    new_arg = arg
                    for pattern, replacement in self.SENSITIVE_PATTERNS:
                        new_arg = re.sub(pattern, replacement, new_arg, flags=re.IGNORECASE)
                    new_args.append(new_arg)
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        
        return True


class CorrelationIdFilter(logging.Filter):
    """Filter that adds correlation ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        record.correlation_id = correlation_id.get() or "N/A"
        return True


class StructuredFormatter(logging.Formatter):
    """JSON-like structured formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured output."""
        # Ensure correlation_id exists
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = correlation_id.get() or "N/A"
        
        # Truncate long messages
        msg = record.getMessage()
        if len(msg) > 1000:
            msg = msg[:997] + "..."
        
        # Build structured log entry
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "correlation_id": record.correlation_id,
            "message": msg,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "correlation_id",
            ]:
                log_entry[key] = value
        
        # Format as simple key=value pairs for readability
        parts = [f"{k}={v}" for k, v in log_entry.items()]
        return " | ".join(parts)


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    structured: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom format string (optional)
        structured: Use structured JSON-like formatting
        log_file: Path to log file (optional)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        format_str = format_str or "%(asctime)s | %(levelname)s | %(name)s | correlation_id=%(correlation_id)s | %(message)s"
        formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RedactingFilter())
    console_handler.addFilter(CorrelationIdFilter())
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactingFilter())
        file_handler.addFilter(CorrelationIdFilter())
        root_logger.addHandler(file_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("elasticsearch").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context.
    
    Args:
        cid: Correlation ID (generated if not provided)
        
    Returns:
        The correlation ID
    """
    cid = cid or str(uuid.uuid4())[:8]
    correlation_id.set(cid)
    return cid


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with correlation ID support."""
    return logging.getLogger(name)
