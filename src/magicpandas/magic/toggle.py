from __future__ import annotations
import abc
from magicpandas import util

import warnings

import inspect

import weakref

from abc import abstractmethod, ABC
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
from typing import TypeVar
T = TypeVar('T')

if False:
    from .magic import Magic

"""
todo: maybe just wrap all functions to be toggleable?
but that violates DRY? oyu have to implement it both in the 
toggle ABC and wrap the function
"""


# class Disabled:
#     __self__ = None
#
#     def __init__(self, __func__: Callable):
#         self.__func__ = __func__
#
#     def __bool__(self):
#         return False
#
#     def __repr__(self):
#         return f'disabled {self.__func__}'
#
#     def __get__(self, instance, owner):
#         self.__self__ = util.proxy(instance)
#         return self
#
#     def __call__(self, *args, **kwargs):
#         try:
#             frame = inspect.currentframe().f_back
#             info = inspect.getframeinfo(frame)
#             warnings.warn(f"""
#             {self.__func__} is disabled, but called anyways at
#             {info.filename}:{info.lineno}
#             """)
#         finally:
#             del frame
#         return self.__func__(*args, **kwargs)


class Toggle(abc.ABC):
    def __init__(self, val: bool):
        self.bool = val

    def __call__(self, func: T) -> T:
        """
        enable(magic.Frame.__log__)
        disable(magic.Frame.__log__)
        col = Col()
        enable(col.__log__)
        disable(col.__log__)
        """
        magic = getattr(func, '__self__', None)

        if magic is not None:
            if (
                    isinstance(func, Disabled)
                    and self.bool
            ):
                # enable(col.__log__)
                setattr(magic, func.__func__.__name__, func.__func__)
            elif (
                    not isinstance(func, Disabled)
                    and not self.bool
            ):
                # disable(col.__log__)
                setattr(magic, func.__name__, func)
        else:
            frame = inspect.currentframe()
            """
            enable(magic.Frame.__log__)
            enable(magic.Frame.__from_file__)
            @disable
            def __log__(self):
                ...
            """
            # try:
            #     # (
            #     #     frame
            #     #     .f_back.f_back.f_locals
            #     #     .setdefault('__disabled', set())
            #     #     .__setattr__(func, self.bool)
            #     # )
            # finally:
            #     del frame

            toggle: dict[str, bool] = frame.f_back.f_locals.setdefault('__toggle', {})
            toggle[func.__name__] = self.bool

        return func


enable = Toggle(True)
disable = Toggle(False)


"""
class MagicFrame(magic.Frame):
    enable(magic.Frame.__log__)
    enable(magic.Frame.__from_file__)
    enable(magic.Frame.__align__)
    disable(magic.Frame.__log__)
    
    # toggles for instance
    col = Col()
    disable(col.__log__)
    disable(col.__from_file__)
    disable(col.__align__)
    enable(col.__log__)
    enable(col.__from_file__)
    enable(col.__align__)

    @disable
    def __log__(self):
        ...
"""
