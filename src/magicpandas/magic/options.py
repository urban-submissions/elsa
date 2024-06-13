from __future__ import annotations
import copy
import numpy as np
from numpy import ndarray
from geopandas import GeoDataFrame, GeoSeries
from pandas import IndexSlice as idx, Series, DataFrame, Index, MultiIndex, Categorical, CategoricalDtype
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy
import geopandas as gpd
from functools import *
from typing import *
from types import *
from shapely import *
import magicpandas as magic

import collections
from types import *

if False:
    from .magic import Magic


class Options:
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__cache__ = {}

    def __get__(
            self,
            instance,
            owner: type[Magic],
    ) -> MappingProxyType[str, bool]:
        from magicpandas.magic.magic import Magic
        if owner not in self.__cache__:
            result = {
                key: value
                for base in owner.__bases__[::-1]
                if issubclass(base, Magic)
                for key, value in base.__options__.items()
            }
            try:
                from_options = owner.__dict__['from_options']
            except KeyError:
                ...
            else:
                kwdefaults = from_options.__func__.__kwdefaults__
                if kwdefaults is None:
                    raise NotImplementedError
                result.update(kwdefaults)
                # result.update(from_options.__kwdefaults__)

                if not isinstance(from_options, classmethod):
                    raise ValueError(
                        f"{owner.__module__}.{owner.__name__}.from_options"
                        f" must be a classmethod"
                    )

                # noinspection PyUnresolvedReferences
                if from_options.__func__.__code__.co_argcount > 1:
                    raise ValueError(f"""
                    {owner.__module__}.{owner.__name__}.from_options
                    must not have any positional arguments!
                    be sure this method looks like this, with the
                    'cls', *, and **kwargs:
                    @classmethod
                    def from_options(cls, *, ..., **kwargs):
                        ...
                    """)
            result = MappingProxyType(result)
            self.__cache__[owner] = result
        else:
            result = self.__cache__[owner]
        return result



class Options(collections.UserDict):
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__cache__ = {}

    def __get__(
            self,
            instance,
            owner: type[Magic],
    ) -> MappingProxyType[str, bool]:
        from magicpandas.magic.magic import Magic

        if owner not in self.__cache__:
            result = self.copy()
            result.update({
                key: value
                for base in owner.__bases__[::-1]
                if issubclass(base, Magic)
                for key, value in base.__options__.items()
            })
            try:
                from_options = owner.__dict__['from_options']
            except KeyError:
                ...
            else:
                kwdefaults = from_options.__func__.__kwdefaults__
                if kwdefaults is None:
                    raise NotImplementedError
                result.update(kwdefaults)

            result = MappingProxyType(result)
            self.__cache__[owner] = result
        else:
            result = self.__cache__[owner]
        return result
