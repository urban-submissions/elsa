from logging import FileHandler, StreamHandler, LogRecord
from copy import copy
from pathlib import Path

class UnindentedFileHandler(FileHandler):
    def __init__(self, *args, **kwargs):
        path = Path( __file__, '..', 'logs', 'log.log').resolve()
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch(exist_ok=True)
        args = (path.__str__(), 'w')
        super(UnindentedFileHandler, self).__init__(*args, **kwargs)

    def emit(self, record):
        new = copy(record)
        msg = record.msg.lstrip('\t')
        msg = msg.lstrip(' ')
        new.msg = msg
        super().emit(new)

class SameLineStreamHandler(StreamHandler):

    def emit(self, record: LogRecord) -> None:
        new = copy(record)
        msg = record.msg.replace('\n', '')
        new.msg = msg
        super().emit(new)
