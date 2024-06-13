from __future__ import annotations
import copy
import inspect
from magicpandas.magic.abc import ABCMagic
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import cached_property, wraps, lru_cache, singledispatch
from typing import *
from types import *
from logging import Logger

if False:
    from .magic import Magic


# todo: not tested yet

# todo: raise Exception if Magic wraps log or log wraps Magic

class _Log:
    """
    @magic.log.debug
    def func():
        ...

    @magic.log.info('blah blah blah')
    def func():
        ...
    """
    log_func = None
    __func__: MethodDescriptorType | FunctionType = None
    __Self__: type[Magic] = None
    __self__: Magic = None




    def __getattr__(self, item):
        return (
            self
            .__func__
            .__get__(self.__self__, self.__Self__)
            .__getattribute__(item)
        )

    @property
    def log_args(self) -> list:
        # return self.__dict__['args']
        return self.__dict__.get('args', [])

    @log_args.setter
    def log_args(self, value):
        if (
                len(value) == 1
                and len(self.log_kwargs) == 0
                and inspect.isfunction(value[0])
        ):
            """
            @magic.log.debug
            def func():
                ...
            """
            self.__func__ = value[0]

        elif (
                len(value) == 1
                and len(self.log_kwargs) == 0
                and isinstance(value[0], ABCMagic)
        ):
            """
            @magic.log.debug
            @Magic
            def func():
                ...
            """
            magic: Magic = value[0]
            log = self.__class__.__name__
            # noinspection PyTypeChecker
            magic: str = magic.__class__.__name__
            raise ValueError(f"""
            {log} was used to wrap an instance of {magic}; 
            Instead, {magic} should be used to wrap {log}.
            """)

        else:
            """
            @magic.log.debug('blah blah blah')
            def func():
                ...
            """
            self.__dict__['args'] = list(value)

    @property
    def msg(self):
        if 'msg' in self.log_kwargs:
            return self.log_kwargs['msg']
            # return self.kwargs.pop('msg')
        elif self.log_args:
            return self.log_args.pop(0)
        else:
            ...

    def __init__(self, log: MethodDescriptorType | Callable, *args, **kwargs):
        self.log_func = log
        self.log_kwargs = kwargs
        self.log_args = args

    def __get__(self, instance: Magic, owner: type[Magic]) -> Self:
        result = copy.copy(self)
        result.__self__ = instance
        result.__Self__ = owner
        return result

    def __call__(self, *args, **kwargs):
        if self.__self__ is None:
            """
            @magic.log.debug('blah blah blah')
            def func():
                ...
            """
            self.log_args = args
            self.log_kwargs = kwargs
            return

        if self.__func__ is None:
            raise ValueError('No function to log')

        # kwargs = self.log_kwargs.copy()
        # args = list(self.log_args)
        logkwargs = self.log_kwargs.copy()
        logargs = list(self.log_args)
        if 'msg' in self.log_kwargs:
            msg = logkwargs.pop('msg')
        elif logargs:
            msg = logargs.pop(0)
        else:
            msg = f'{self.__self__.__trace__}'

        logger = self.__self__.__logger__
        (
            self.log_func
            .__get__(logger, logger.__class__)
            .__call__(msg=msg, *logargs, **logkwargs)
        )
        result = (
            self
            .__func__
            .__get__(self.__self__, self.__Self__)
            .__call__(*args, **kwargs)
        )
        return result


T = TypeVar('T')
class Log:
    default = Logger.debug

    def debug(self, msg: str | T) -> T:
        return _Log(Logger.debug, msg)

    def info(self, msg: str | T) -> T:
        return _Log(Logger.info, msg)

    def warning(self, msg: str | T) -> T:
        return _Log(Logger.warning, msg)

    def error(self, msg: str | T) -> T:
        return _Log(Logger.error, msg)

    def critical(self, msg: str | T) -> T:
        return _Log(Logger.critical, msg)

    def __call__(self, msg: str | T) -> T:
        return _Log(self.debug, msg)


log = Log()


