from __future__ import annotations

import contextlib
import shutil
from typing import Callable, TypeVar, Any, Union

import functools
import threading
from functools import wraps
import logging
from logging import Logger
from typing import Type

if False:
    from magicpandas.frame import Frame

"""
DEBUG    									missing_nodes.coords.matches.from_self		coords.matches.Matches
DEBUG    											missing_nodes.coords.edges.is_loop		edges.Edges
DEBUG    									missing_nodes.coords.result.nodes.as_tuples		nodes.Nodes
DEBUG    									missing_nodes.coords.result.nodes.iend_tuple		nodes.Nodes
DEBUG    									missing_nodes.coords.result.edges.iend_iend		edges.Edges
DEBUG    									missing_nodes.coords.result.edges.is_loop		edges.Edges

todo: perhaps whitespace the repetitive parts of the traceback
"""


class Local(threading.local):
    _instance = None

    def __init__(self):
        self.level = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Local, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class Wrap:
    local = Local()
    owner = None
    Owner = None

    def __repr__(self):
        repr = (
                self.fget.__repr__()
                + f' in {self.__class__.__qualname__}'
        )
        return repr

    @property
    def __name__(self):
        return self.fget.__name__

    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, func, log, logger):
        self.fget = func
        self.log = log
        self.logger = logger

    def __get__(self, instance: Frame, owner: Type[Frame]) -> Union[Wrap, Callable]:
        assert not isinstance(instance, type)
        self.owner = instance
        self.Owner = owner
        if instance is not None:
            return functools.partial(self, instance)

        return self

    def __call__(self, *args: Frame, **kwargs):
        indent = '    ' * self.logger.local.level
        module = self.fget.__module__
        binding = self.fget.__qualname__.rsplit('.', 1)[0]
        owner = args[0]
        if owner.magic.message is None:
            traceback = owner.__trace__
            if traceback:
                traceback += '.'
            traceback += self.fget.__name__
        else:
            traceback = owner.magic.message

        # todo: fix indentation on rjustified segment
        #   seems like there's unintended tabs on the left side when indented
        # total_length = 80
        # todo: why come shutil.get_terminal_size always using fallback?
        total_length, _ = shutil.get_terminal_size(fallback=(80, 24))
        message = f'{indent}{traceback}'
        right_segment = f'{module}.{binding}'
        spaces_needed = total_length - (len(message) + len(right_segment))
        # msg = f'{message}{" " * spaces_needed}{right_segment}'
        msg = f"{message}\t\t{right_segment}"
        self.log(msg)
        self.logger.local.level += 1
        # if self.fget.__qualname__ == 'Coords.from_nested':
        #     raise AttributeError
        result = self.fget(*args, **kwargs)
        # todo: how do we support self.from_root(owner)?

        self.logger.local.level -= 1
        return result

    """
    @magic.attr
    @magic.console.wrap.debug
    def test(self):
        in this case, magic.attr passes self.instance to self.fget.__call__, works
        
    
    @magic.logger.console.wrap.debug
    def from_graph(self, owner: Graph) -> Coords:
        in this case, Frame.__get__ calls from_graph(self, owner), works but might need to be changed
        should return partial in __get__ if owner not None?
        
    @magic.logger.console.wrap.debug
    def from_graph(self, owner: Graph) -> Coords:
        in this case, debug.__call__ calls from_graph(owner), fails
        should return partial? 
    """


class WrapFactory:
    instance: IndentationLogger

    def __get__(self, instance: IndentationLogger, owner: Type[IndentationLogger]) -> WrapFactory:
        self.owner = instance
        self.Owner = owner
        return self

    def info(self, func):
        return Wrap(func, self.owner.info, self.owner)

    def warning(self, func):
        return Wrap(func, self.owner.warning, self.owner)

    def error(self, func):
        return Wrap(func, self.owner.error, self.owner)

    def critical(self, func):
        return Wrap(func, self.owner.critical, self.owner)

    def debug(self, func):
        return Wrap(func, self.owner.debug, self.owner)


def add_indent(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(
            self,
            msg: Any,
            *args: Any,
            exc_info: _ExcInfoType = None,
            stack_info: bool = False,
            stacklevel: int = 1,
            extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        indent = '\t' * self.local.level
        # todo: not sure why extra indent necessary
        msg = f"{indent}\t{msg}"
        func(self, msg, *args, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)

    return wrapper


class IndentationLogger(Logger):
    wrap = WrapFactory()
    local = Local()
    info = add_indent(Logger.info)
    warning = add_indent(Logger.warning)
    error = add_indent(Logger.error)
    critical = add_indent(Logger.critical)
    debug = add_indent(Logger.debug)

    # @contextlib.contextmanager
    # def silence(self, level: int = logging.ERROR):
    #     previous = self.getEffectiveLevel()
    #     self.setLevel(level)
    #     try:
    #         yield
    #     finally:
    #         self.setLevel(previous)
