from __future__ import annotations
from pandas.core.generic import is_list_like
from typing import *

from magicpandas.magic.magic import Magic
from magicpandas.pandas.cached import cached
from magicpandas.pandas.ndframe import NDFrame
from magicpandas.pandas.series import Series

import pandas as pd

import magicpandas
from magicpandas.pandas.column import Column
from magicpandas.magic.magic import Magic

if False:
    from magicpandas import Frame




class Index(Column):
    @cached.diagonal.property
    def __permanent__(self):
        return True

    def __subget__(self: Column, outer: Frame, Outer) -> Column:
        owner: Frame = self.__owner__
        key = self.__key__
        if self.__permanent__:
            owner.__permanent__.add(key)
        owner.__columns__.add(key)
        try:
            return owner.index.get_level_values(key)
        except KeyError:
            ...
        try:
            return super().__subget__(outer, Outer)
        except KeyError:
            ...

        try:
            result = (
                self
                .__from_outer__
                .__func__
                .__get__(outer, type(outer))
                .__call__()
            )
        except NotImplementedError as e:
            trace = self.__owner__.__trace__
            raise AttributeError(
                f'`{key}` not found in frame `{trace}` and no constructor '
                f'found for {self.__class__.__qualname__} instance `{self.__trace__}`'
            ) from e

        owner[key] = result
        owner.set_index(key, inplace=True)
        result = owner.index.get_level_values(key)
        return result

    def __subset__(self, instance, value):
        key = self.__key__
        # match getattr(instance, key):
        #     case pd.Series():
        #         instance[key] = value
        #         # super(Index, self).__subset__(instance, value)
        #
        #     case pd.Index():
        #         value = instance.index = instance.index.set_levels(value, level=key)
        #     case _:
        #         raise TypeError(f'cannot set {key} to {type(value)}')

        if key == instance.index.name:
            instance.index = value
        elif key in instance.index.names:
            instance.index = instance.index.set_levels(value, level=key)
        else:
            # column subset
            super().__subset__(instance, value)

        return value

    def __subget__(self: Column, outer: Frame, Outer) -> Column:
        # assert self.__owner__.__order__ == 3
        key = self.__key__
        owner: Frame = self.__owner__

        if self.__permanent__:
            owner.__permanent__.add(key)
        owner.__columns__.add(key)

        # todo: problem is that accessing while it's a copy will perform setitem
        if key in owner._item_cache:
            # noinspection PyTypeChecker
            return owner._item_cache[key]
        if key in owner:
            result = owner._get_item_cache(key)
            return result
        # elif key in owner.index.names:
        elif key == owner.index.name:
            return owner.index
        elif key in owner.index.names:
            result = owner.index.get_level_values(key)
            return result
        elif self.__from_outer__:
            func = self.__from_outer__.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            result = self.__log__(func)
            result = self.__postprocess__(result)
            outer.__inner__ = inner
            result = self.__subset__(owner, result)
        elif self.__from_inner__:
            result = self.__log__(self.__from_inner__)
            result = self.__postprocess__(result)
            result = self.__subset__(owner, result)
        else:
            raise NotImplementedError(
                f'Could not resolve a constructor for {self.__trace__}'
            )

        return result


def index(func) -> Union[pd.Series, pd.Index]:
    ...


globals()['index'] = Index

"""
Todo: we need to allow for mapping of old index to new index
for magicpandas NDFrame set_axis to not ruin the frames and columns stored in .attrs

DataFrame.

"""
