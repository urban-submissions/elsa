from __future__ import annotations
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

# from pandas import is_list_like
from pandas.core.generic import is_list_like
from typing import *

from magicpandas.magic.magic import Magic
from magicpandas.pandas.cached import cached
from magicpandas.pandas.ndframe import NDFrame
from magicpandas.pandas.series import Series
from typing import TypeVar

T = TypeVar('T')

if False:
    from .frame import Frame


class Column(Series):
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]
    __init__ = Series.__init__

    def __subget__(self: Column, outer: Frame, Outer) -> Column:
        # assert self.__owner__.__order__ == 3
        owner = self.__owner__
        key = self.__key__

        if self.__permanent__:
            owner.__permanent__.add(key)
        owner.__columns__.add(key)

        if key in owner._item_cache:
            # noinspection PyTypeChecker
            return owner._item_cache[key]
        if key in owner:
            # result = owner._get_item_cache(key)
            try:
                result = owner._get_item_cache(key)
            except TypeError:
                # TypeError: only integer scalar arrays can be converted to a scalar index
                # During handling of the above exception, another exception occurred:
                result = owner[key]
            if owner._is_copy:
                result = self(result)
                owner._item_cache[key] = result
                # if (
                #         len(owner) != len(result)
                #         or len(outer) != len(result)
                # ):
                #     raise ValueError

                return result
            if not isinstance(result, self.__class__):
                result = self.__subset__(owner, result)
                # if (
                #         len(owner) != len(result)
                #         or len(outer) != len(result)
                # ):
                #     raise ValueError
                #
        elif self.__from_outer__:
            func = self.__from_outer__.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            with self.__recursion__():
                result = self.__log__(func)
            result = self.__postprocess__(result)
            outer.__inner__ = inner
            result = self.__subset__(owner, result)
            # if (
            #         len(owner) != len(result)
            #         or len(outer) != len(result)
            # ):
            #     raise ValueError
        elif self.__from_inner__:
            # result: Column = self.__log__(self.__from_inner__)
            with self.__recursion__():
                result = self.__log__(self.__from_inner__)
            result = self.__postprocess__(result)
            result = self.__subset__(owner, result)
        else:
            msg = (
                f'Could not resolve a constructor for {self.__trace__}. '
                f'If attempting to lazy-compute the object, please '
                f'assure the method returns something. Otherwise, the '
                f'object is being get before it is set.'
            )
            raise NotImplementedError(msg)

        # if (
        #         len(owner) != len(result)
        #         or len(outer) != len(result)
        # ):
        #     raise ValueError
        del result.__trace__
        _ = result.__trace__
        return result

    def __subset__(self, instance, value):
        owner = self.__owner__
        key = self.__key__
        owner[key] = value
        sliced, owner._sliced_from_mgr = owner._sliced_from_mgr, self._from_mgr
        # result = owner._get_item_cache(key)
        try:
            result = owner._get_item_cache(key)
        except TypeError:
            # TypeError: only integer scalar arrays can be converted to a scalar index
            # During handling of the above exception, another exception occurred:
            result = owner[key]
        # result = self(result)
        # result = self(result).astype(self.__dtype__)
        result = self(result)
        dtype = self.__dtype__
        if dtype is not None:
            result = result.astype(dtype)
        # result.__third__ = result
        owner._item_cache[key] = result
        owner._sliced_from_mgr = sliced
        if len(value) != len(owner):
            raise ValueError
        if len(owner) != len(result):
            raise ValueError
        return result

    def __subset__(self, owner, value):
        key = self.__key__
        owner[key] = value
        sliced, owner._sliced_from_mgr = owner._sliced_from_mgr, self._from_mgr
        try:
            result = owner._get_item_cache(key)
        except TypeError:
            # i think this occurs when index is a RangeIndex?
            result = owner[key]
        # result = self(result)
        # result = self(result).astype(self.__dtype__)
        result = self(result)
        dtype = self.__dtype__
        if dtype is not None:
            result = result.astype(dtype)
        # result.__third__ = result
        owner._item_cache[key] = result
        owner._sliced_from_mgr = sliced
        # if (
        #         # self.__name__ == 'file'
        #         result.isna().any()
        # ):
        #     raise ValueError
        return result

    def __subdelete__(self, instance):
        super().__subdelete__(instance)
        try:
            del self.__owner__[self.__key__]
        except KeyError:
            ...

    @cached.base.property
    def __flush__(self):
        return True

    # @cached.base.property
    @cached.diagonal.property
    def __permanent__(self):
        """
        If True, calling del on the attr will do nothing; this is
        for when the column should not be flushed by flush_columns
        """
        return False

    # @cached.base.property
    @cached.diagonal.property
    def __postinit__(self):
        """
        If True, the column will be initialized after the initialization
        of the owner, rather than needing to be accessed first.
        """
        return False

    def __postprocess__(self, result: T) -> T:
        owner = self.__owner__
        # if isinstance(result, pd.Series) and (
        #     result.index.name == owner.index.name
        #     or result.index.names == owner.index.names
        # ):
        #     result = result.reindex(owner.index)
        #


        if (
                is_list_like(result)
                and len(result) != len(owner)
        ):
            raise ValueError(
                f'{self.__trace__} returned a list-like with length '
                f'{len(result)}, but owner has length {len(owner)}.'
            )

        if isinstance(result, Series):
            result = result.values
        result = self(result, index=owner.index, name=self.__key__, dtype=self.__dtype__)
        return super().__postprocess__(result)

    @classmethod
    def from_options(
            cls,
            *,
            log=True,
            from_file=False,
            align=False,
            dtype=None,
            permanent=False,
            postinit=False,
            no_recursion=False,
            **kwargs,
    ):
        result: Self = super().from_options(
            log=log,
            from_file=from_file,
            align=align,
            dtype=dtype,
            permanent=permanent,
            postinit=postinit,
            no_recursion=no_recursion,
            **kwargs,
        )
        return result

    def __set_name__(self, owner: type[Magic], name):
        super().__set_name__(owner, name)
        if self.__postinit__:
            if not issubclass(owner, NDFrame):
                raise ValueError(
                    f"Currently {Column.__name__}.__postinit__ is only "
                    f"supported for {NDFrame.__name__} subclasses. "
                    f"Owner is {owner.__class__.__name__}."
                )
            owner.__postinits__.add(name)


if __name__ == '__main__':
    from magicpandas.pandas.frame import Frame
