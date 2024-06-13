from __future__ import annotations

import pandas.core.generic
import copy
import inspect
import os
import warnings
import weakref
from functools import cached_property
from typing import *

import numpy as np
import pandas as pd
import pandas.core.common
from pandas import DataFrame
from pandas import MultiIndex
from pandas.core.generic import (
    ABCDataFrame,
    is_list_like,
    find_stack_level,
)
from pandas.core.groupby import DataFrameGroupBy

import magicpandas.pandas.cached
from magicpandas.magic.classboundstrings import ClassBoundStrings
from magicpandas.magic.default import default
from magicpandas.magic.magic import Magic
from magicpandas.magic.truthy import truthy
from magicpandas.pandas.attrs import Attrs
from magicpandas.pandas.cached import cached
from magicpandas.magic.order import Order
from typing import TypeVar
T = TypeVar('T')

if False:
    from .series import Series


def __setattr__(self, name: str, value) -> None:
    """
    If it can be found nowhere else than the info_axis, it must be a column.
    This allows simpler access to columns for interactive use.
    """
    if (
            name not in self.__dict__
            and name not in self._internal_names_set
            and name not in self._metadata
            and name in self._info_axis
    ):
        try:
            self[name] = value
        except (AttributeError, TypeError):
            pass
        else:
            return

    if (
            # only relevant with dataframes
            isinstance(self, ABCDataFrame)
            # ignore if attr already exists
            and name not in self.__dict__
            # ignore if it's a class attribute;
            # might be a prop, axis, cached_property,
            # or other sort of descriptor
            and not hasattr(type(self), name)
            # ignore if internal or metadata
            and name not in self._internal_names
            and name not in self._metadata
            and is_list_like(value)
    ):
        warnings.warn(
            "Pandas doesn't allow columns to be "
            "created via a new attribute name - see "
            "https://pandas.pydata.org/pandas-docs/"
            "stable/indexing.html#attribute-access",
            stacklevel=find_stack_level(),
        )
    object.__setattr__(self, name, value)


# monkey patch my pandas.DataFrame bugfix for now;
# otherwise DataFrame.__setattr__ causes recursion
# see https://github.com/pandas-dev/pandas/pull/56794
pandas.core.generic.NDFrame.__setattr__ = __setattr__


def apply_if_callable(maybe_callable, obj, **kwargs):
    """
    Evaluate possibly callable input using obj and kwargs if it is callable,
    otherwise return as it is.

    Parameters
    ----------
    maybe_callable : possibly a callable
    obj : NDFrame
    **kwargs
    """
    # modified to also check if is_list_like
    if (
            callable(maybe_callable)
            and not is_list_like(maybe_callable)
    ):
        return maybe_callable(obj, **kwargs)

    return maybe_callable


pandas.core.common.apply_if_callable = apply_if_callable


class SubLoc:
    """
    Index an NDFrame on a level of its MultiIndex.
    This requires that the index level is unique.
    """
    instance: Series | DataFrame = None

    def __get__(self, instance, owner) -> Self:
        if instance is None:
            return None
        result = copy.copy(self)
        result.instance = instance
        return result

    def __getitem__(self, item):
        loc : Series | np.ndarray
        if isinstance(item, tuple):
            loc, cols = item
        else:
            loc = item
            cols = slice(None)
        name = getattr(loc, 'name', None)
        if not name:
            raise ValueError(f'Cannot use {loc} as a key')
        instance = self.instance
        attr = f'{self.__name__}.{name}'
        if attr not in instance.__dict__:
            index = instance.index.get_level_values(name)
            if index.duplicated().any():
                raise ValueError(
                    f'Cannot subloc on {name} because it is not unique'
                )
            data = np.arange(len(index))
            iloc = pd.Series(data, index=index)
            instance.__dict__[attr] = iloc
        else:
            iloc = instance.__dict__[attr]
        iloc = iloc.loc[loc]
        if isinstance(cols, slice):
            result = instance.iloc[iloc]
        elif isinstance(cols, Iterable):
            icol = [instance.columns.get_loc(col) for col in cols]
            result = instance.iloc[iloc, icol]
        else:
            icol = instance.columns.get_loc(cols)
            result = instance.iloc[iloc, icol]
        return result


    def __setitem__(self, item, value):
        loc : Series | np.ndarray
        if isinstance(item, tuple):
            loc, cols = item
        else:
            loc = item
            cols = slice(None)
        name = getattr(loc, 'name', None)
        if not name:
            raise ValueError(f'Cannot use {loc} as a key')
        instance = self.instance
        attr = f'{self.__name__}.{name}'
        if attr not in instance.__dict__:
            index = instance.index.get_level_values(name)
            if index.duplicated().any():
                raise ValueError(
                    f'Cannot subloc on {name} because it is not unique'
                )
            data = np.arange(len(index))
            iloc = pd.Series(data, index=index)
            instance.__dict__[attr] = iloc
        else:
            iloc = instance.__dict__[attr]
        iloc = iloc.loc[loc]
        if isinstance(cols, slice):
            instance.iloc[iloc] = value
        elif isinstance(cols, Iterable):
            icol = [instance.columns.get_loc(col) for col in item[1:]]
            instance.iloc[iloc, icol] = value
        else:
            icol = instance.columns.get_loc(cols)
            instance.iloc[iloc, icol] = value


    def __set_name__(self, owner, name):
        self.__name__ = name


# todo: we need to be able to call __set__ without another wrapper recursion

class NDFrame(
    magicpandas.pandas.cached.Magic,
    pandas.core.generic.NDFrame,
    from_outer=True,
):
    __inner__: pd.Series | pd.DataFrame | NDFrame | Any
    __inner__: Magic
    __outer__: Magic
    __futures__ = cached.root.property(Magic.__futures__)
    __threads__ = cached.root.property(Magic.__threads__)
    __done__ = cached.root.property(Magic.__done__)
    # __rootdir__ = cached.root.property(Magic.__rootdir__)
    __postinits__ = ClassBoundStrings()
    __init_nofunc__ = pandas.core.generic.NDFrame.__init__
    loc: MutableMapping[Any, Self] | Self
    iloc: MutableMapping[Any, Self] | Self
    # attrs = Attrs()   # likely delete this because it causes problems for to_parquet
    __direction__ = 'horizontal'
    __sticky__ = False
    subloc = SubLoc()
    subloc: MutableMapping[Any, Self] | Self

    def __subget__(self, outer: Magic, Outer) -> NDFrame:
        result: NDFrame
        if outer is None:
            return self

        owner: NDFrame | Series | DataFrame = self.__owner__
        key = self.__key__
        if self.__configuring__:
            return self
        elif self.__from_params__:
            return self
        elif default.context:
            return self
        elif (
            outer is not None
            and outer.__max__ < 3
        ):
            return self
        if key in owner.__dict__:
            return owner.__dict__[key]

        # todo problem is we get here, and second gets new owner
        #   todo probably not propagate every time accessing second
        _ = self.__second__
        volatile = self.__volatile__.copy()

        if (
                key in owner.attrs
                and self.__align__
        ):
            # subindex from already existing frame
            result: Self = owner.attrs[key]
            result = result.__align__(owner)
        elif (
                isinstance(owner, pd.DataFrame)
                and isinstance(self, pd.Series)
                and key in owner.columns
        ):
            # noinspection PyTypeChecker
            result = self(owner[key])
        elif (
                self.__from_file__
                and os.path.exists(self)
        ):
            result = self.__log__(self.__from_file__)
            result = self.__postprocess__(result)
        elif self.__from_outer__:
            func = self.__from_outer__.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            result = self.__log__(func)
            outer.__inner__ = inner
            result = self.__postprocess__(result)
            if (
                    self.__from_file__
                    and not os.path.exists(self)
            ):
                self.__to_file__(result)
        elif self.__from_inner__:
            # load from inner
            result = self.__log__(self.__from_inner__)
            result = self.__postprocess__(result)
            if (
                    self.__from_file__
                    and not os.path.exists(self)
            ):
                self.__to_file__(result)
        else:
            raise NotImplementedError(
                f'Could not resolve a constructor for {self.__trace__}'
            )

        result = self.__subset__(outer, result)
        result.__volatile__.update(volatile)
        assert isinstance(result, self.__class__)
        # todo: this seems like a suboptimal solution;
        #   the problem is trace is horizontal, while self is 2nd order
        #   and result is 3rd order
        del result.__trace__
        _ = result.__trace__
        return result

    def __propagate__(self, obj: NDFrame | T) -> T:
        """ set metadata from another object """
        # todo: _ = self.second resets owner?
        if self.__order__ == 2:
            _ = self.__second__
        if obj.__order__ == self.__order__:
            # cache = self.__horizontal__
            cache = self.__directions__.horizontal
        elif obj.__order__ > self.__order__:
            # cache = self.__vertical__
            cache = self.__directions__.vertical
        else:
            raise ValueError(f'obj.order < self.order')
        diagonal = self.__directions__.diagonal
        obj.attrs.update(
            (key, value)
            for key, value in self.attrs.items()
            if key in cache
            or key in diagonal
        )
        obj.__dict__.update(
            (key, value)
            for key, value in self.__dict__.items()
            if key in cache
            or key in diagonal
        )
        obj.__volatile__.update(self.__volatile__)
        return obj

    # def __propagate__(self, obj: NDFrame | T) -> T:
    #     """ set metadata from another object """
    #     if self.__order__ == 2:
    #         _ = self.__second__
    #     propagating = self.__propagating__
    #     if obj.__order__ == self.__order__:
    #         propagating = propagating.horizontal
    #     elif obj.__order__ > self.__order__:
    #         propagating = propagating.vertical
    #     else:
    #         raise ValueError(f'obj.order < self.order')
    #     if obj.__trace__ != self.__trace__:
    #         propagating = propagating.sticky
    #     obj.__cache__.update(
    #         (key, value)
    #         for key, value in self.__cache__.items()
    #         if key in propagating
    #     )
    #     obj.__dict__.update(
    #         (key, value)
    #         for key, value in self.__dict__.items()
    #         if key in propagating
    #     )
    #     obj.__volatile__.update(self.__volatile__)
    #     return obj

    @property
    def __cache__(self):
        return self.attrs

    def __call__(self, *args, **kwargs) -> Self:
        if self.__from_params__:
            result = self.__from_params__(*args, **kwargs)
            self.__propagate__(result)
            return result
        result = self.__constructor__(*args, **kwargs)
        return result

    # 1. propagate from frame to result
    # 2. propagate from result to frame

    def __constructor__(self, *args, **kwargs) -> Self:
        if args:
            frame = args[0]
            if (
                    isinstance(frame, NDFrame)
                    and frame.__order__ != 3
            ):
                warnings.warn(f"""
                {self.__trace__} is being called on a frame with order 
                {frame.__order__}, which is likely unintended. Are you 
                calling on `self.__outer__` instead of `self.__owner__`?
                """)

            if inspect.isfunction(frame):
                self.__init_func__(frame, *args, **kwargs)
                return self
            result = self.__class__(*args, **kwargs)

            # todo: test this: this preserves attrs like Frame.copy
            #   but also when calling self.__inner__ so we don't
            #   just clobber all cached attributse
            # if isinstance(frame, pandas.core.generic.NDFrame):
            #     result.attrs.update(frame.attrs)
            # if (
            #     isinstance(frame, NDFrame)
            #     and frame.__order__ == 3
            # ):


            # sticky or same trace
            if (
                    isinstance(frame, NDFrame)
                    and frame.__trace__ == self.__trace__
            ):
                frame.__propagate__(result)
            if not inspect.isfunction(frame):
                self.__propagate__(result)
        else:
            result = self.__class__(*args, **kwargs)
            self.__propagate__(result)
        return result

    # def __constructor__(self, *args, **kwargs) -> Self:
    #     if args:
    #         frame = args[0]
    #         if inspect.isfunction(frame):
    #             self.__init_func__(frame, *args, **kwargs)
    #             return self
    #         if (
    #                 isinstance(frame, NDFrame)
    #                 and frame.__order__ != 3
    #         ):
    #             warnings.warn(f"""
    #             {self.__trace__} is being called on a frame with order
    #             {frame.__order__}, which is likely unintended. Are you
    #             calling on `self.__outer__` instead of `self.__owner__`?
    #             """)
    #
    #         result = self.__class__(*args, **kwargs)
    #         if isinstance(frame, NDFrame):
    #             frame.__propagate__(result)
    #
    #         self.__propagate__(result)
    #     else:
    #         result = self.__class__(*args, **kwargs)
    #         self.__propagate__(result)
    #     return result

    def __flush_references(self) -> Self:
        result = self.copy()
        dropping = {
            key
            for key, value in self.__cache__.items()
            if isinstance(value, weakref.ReferenceType)
        }
        dropping.update(
            key
            for key, value in self.__dict__.items()
            if isinstance(value, weakref.ReferenceType)
        )
        if dropping:
            warnings.warn(
                f'{self.__trace__} is being aligned but contains weakreferences {dropping}'
            )
        result.attrs = {
            key: value
            for key, value in self.attrs.items()
            if not isinstance(value, weakref.ReferenceType)
        }
        result.__dict__ = {
            key: value
            for key, value in self.__dict__.items()
            if not isinstance(value, weakref.ReferenceType)
        }
        return result

    @truthy
    def __align__(self, owner: NDFrame = None) -> Self:
        """ align such that self.index ⊆ outer.index """
        if not self.__align__:
            return self.copy()
        if owner is None:
            owner = self.__owner__
        result = (
            self
            .__subalign__(owner)
            .copy()
        )
        result.__owner__ = owner
        return result

    # todo: we need to modify subalign so that owner is passed;
    #   we cannot use self.__owner__ because the reference might be lost

    def __subalign__(self, owner: NDFrame) -> Self:
        """ align such that self.index ⊆ outer.index """
        haystack: MultiIndex = owner.index
        needles: MultiIndex = self.index
        if (
                isinstance(needles, pd.Index)
                and not needles.name
        ):
            # If index unnamed, assume it is the same as the owner's index
            ...
        elif set(needles.names).intersection(haystack.names):
            # If some index names are shared, align on those
            try:
                names = haystack.names.difference(needles.names)
                haystack = haystack.droplevel(names)
                names = needles.names.difference(haystack.names)
                needles = needles.droplevel(names)
            except ValueError as e:
                raise ValueError(
                    f'Could not align {self.__trace__} with {owner.__trace__};'
                    f' be sure that the index names are compatible.'
                ) from e
        elif (
                isinstance(self, DataFrame)
                and set(haystack.names).intersection(self.columns)
        ):
            # If some columns are in the owner's index names, align on those
            columns = set(haystack.names).intersection(self.columns)
            needles = pd.MultiIndex.from_frame(self[columns])
        else:
            raise NotImplementedError(
                f'Could not resolve how to align {self.__trace__} with '
                f'owner {self.__owner__.__trace__}; you may define how '
                f'to align by overriding {self.__class__}.__align__.'
            )

        loc = needles.isin(haystack)
        result = self.loc[loc]
        return result

    def __subset__(self, outer: NDFrame, value: NDFrame):
        if self.__configuring__:
            raise NotImplementedError
        owner: NDFrame = self.__owner__
        key = self.__key__
        result = self.__call__(value)

        if self.__align__:
            result = result.__align__(owner)
            owner.attrs[key] = result
        # print(f'putting {key=} into {owner.__trace__=}')
        owner.__dict__[key] = result

        for postinit in self.__postinits__:
            if postinit in self:
                continue
            getattr(result, postinit)

        return result

    def __subdelete__(self, outer: NDFrame):
        if self.__configuring__:
            raise NotImplementedError
        owner = self.__owner__
        key = self.__key__
        if key in owner.__dict__:
            del owner.__dict__[key]
        if key in owner.attrs:
            del owner.attrs[key]

    @cached_property
    def _constructor(self):
        return type(self)

    @classmethod
    def from_options(
            cls,
            *,
            postinit=False,
            log=True,
            from_file=False,
            align=False,
            **kwargs
    ) -> Callable[[...], Self]:
        return super().from_options(
            postinit=postinit,
            log=log,
            from_file=from_file,
            align=align,
            **kwargs
        )


