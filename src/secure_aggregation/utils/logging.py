import logging
import sys
from logging import Logger
from typing import Optional


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter for log records."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        log = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%SZ"),
        }
        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)
        return str(log)


def configure_logging(level: str = "INFO", json_output: bool = False, log_file: Optional[str] = None) -> None:
    """
    Configure global logging. Uses stdout by default; can additionally tee to a file.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    formatter: logging.Formatter
    if json_output:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=handlers, force=True)


def get_logger(name: str) -> Logger:
    return logging.getLogger(name)
