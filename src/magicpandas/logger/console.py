from __future__ import annotations
import logging
import logging.config
from pathlib import Path

import contextlib
import logging.config

import logging


if False:
    from .logger import ConsoleLogger

class ConsoleFormatter(logging.Formatter):
    logger: ConsoleLogger = None

    def format(self, record: logging.LogRecord):
        original_message = record.msg
        indentation = ' ' * (4 * self.logger.indent)
        record.msg = f"{indentation}{original_message}"
        formatted_message = super().format(record)
        record.msg = original_message  # Reset to avoid affecting other handlers
        return formatted_message
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ConsoleHandler(logging.StreamHandler):
    formatter: ConsoleFormatter

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



