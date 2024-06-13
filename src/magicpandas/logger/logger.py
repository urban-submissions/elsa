from __future__ import annotations
import sys
import logging
import logging.config
from pathlib import Path

import contextlib
from magicpandas.logger.console import ConsoleFormatter, ConsoleHandler
import logging.config

import logging


class ConsoleLogger(logging.Logger):
    # todo: stacklevel instead of indent?
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indent = 0

    @contextlib.contextmanager
    def silence(self, level: int = logging.ERROR):
        previous = self.getEffectiveLevel()
        self.setLevel(level)
        try:
            yield
        finally:
            self.setLevel(previous)



handler = ConsoleHandler(stream=sys.stderr)
formatter = ConsoleFormatter(fmt='%(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)

logging.setLoggerClass(ConsoleLogger)
logger = logging.getLogger('magicpandas')
logging.setLoggerClass(logging.Logger)
handler.logger = logger
formatter.logger = logger
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False

# logger.debug('hello world')

