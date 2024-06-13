from __future__ import annotations

import copy
import pandas
import warnings
import weakref
from pandas import DataFrame
from pandas import Series
from pandas._libs import lib
from typing import *

from magicpandas.magic.magic import Magic
from magicpandas.magic.order import Order
from magicpandas.magic.truthy import truthy
from magicpandas.pandas.cached import cached
from magicpandas.pandas.ndframe import NDFrame

if False:
    from .frame import Frame


class Series(NDFrame, pandas.Series):
    # __init__ = pandas.Series.__init__
    # __subinit__ = pandas.Series.__init__
    __init_nofunc__ = pandas.Series.__init__
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]

    if False:
        # noinspection PyMethodOverriding
        # @staticmethod
        def __call__(
                self,
                data=None,
                index=None,
                dtype: Dtype | None = None,
                name=None,
                copy: bool | None = None,
                fastpath: bool | lib.NoDefault = lib.no_default,
        ) -> Self:
            ...

    @classmethod
    def from_options(
            cls,
            *,
            log=True,
            from_file=False,
            align=False,
            dtype=None,
            **kwargs,
    ) -> Callable[[...], Self]:
        # 
        # func = super().from_options(
        #     log=log,
        #     from_file=from_file,
        #     align=align,
        #     dtype=dtype,
        #     **kwargs,
        # )
        # @wraps(func)
        # def wrapper(*args, **kwargs):
        #     result: Self = func(*args, **kwargs)
        #     result.__dtype__ = dtype
        #     return result
        # return wrapper
        #
        return super().from_options(
            log=log,
            from_file=from_file,
            align=align,
            dtype=dtype,
            **kwargs,
        )


    def __subset__(self, outer: NDFrame, value):
        # instead of __propagate__, uses __call__
        if self.__configuring__:
            raise NotImplementedError
        owner: NDFrame = self.__owner__
        key = self.__key__
        if (
                isinstance(value, NDFrame)
                and value.__order__ != 3
        ):
            warnings.warn(f"""
            {self.__trace__} is being set to a frame with order 
            {value.__order__}, which is likely unintended. Are you 
            setting to `self.__owner__` instead of `self.__outer__`?
            """)
        result = self(value, dtype=self.__dtype__, name=self.__key__)
        result = result.__align__(owner)
        result.__trace__ = self.__trace__
        owner.__dict__[key] = result
        owner.attrs[key] = weakref.ref(result)
        del result.__trace__
        _ = result.__trace__
        # result.__third__ = result

        return result

    def __subdelete__(self, instance: Frame):
        super().__subdelete__(instance)
        owner = self.__owner__
        key = self.__key__.__str__()
        if isinstance(owner, DataFrame):
            try:
                del instance[key]
            except KeyError:
                ...

    # @lru_cache()
    def __repr__(self):
        result = self.__trace__.__str__()
        match self.__order__:
            case Order.first:
                result += ' 1st order'
            case Order.second:
                result += ' 2nd order'
            case Order.third:
                if result:
                    result += '\n'
                result += f'{pandas.Series.__repr__(self)}'

        return result

    # @cached.base.property
    @cached.diagonal.property
    def __dtype__(self):
        """ The Dtype to assign to the Series, if desired. """
        return None


class Column(Series):
    @cached.base.property
    def __permanent__(self):
        """
        If True, calling del on the attr will do nothing; this is
        for when the column should not be flushed by flush_columns
        """
        return False

    @cached.base.property
    def __postinit__(self):
        """
        If True, the column will be initialized after the initialization
        of the owner, rather than needing to be accessed first.
        """
        return False

    @classmethod
    def from_options(
            cls,
            *,
            log=True,
            from_file=False,
            align=True,
            dtype=None,
            permanent=False,
            postinit=False,
            **kwargs,
    ):
        return super().from_options(
            log=log,
            from_file=from_file,
            dtype=dtype,
            align=align,
            permanent=permanent,
            postinit=postinit,
            **kwargs,
        )

    def __subdelete__(self, instance):
        # warnings.warn(f"{self.__trace__} is defined as a permanent column.")
        if self.__permanent__:
            return
        super().__subdelete__(instance)
        try:
            del self.__owner__[self.__key__]
        except KeyError:
            ...
        # del self.__owner__[self.__key__]

    def __set_name__(self, owner: Magic, name):
        super().__set_name__(owner, name)
        if self.__postinit__:
            if not issubclass(owner, NDFrame):
                raise ValueError(
                    f"Currently {Column.__name__}.__postinit__ is only "
                    f"supported for {NDFrame.__name__} subclasses. "
                    f"Owner is {owner.__class__.__name__}."
                )
            owner.__postinits__.add(name)

    def __subset__(self, instance, value: pandas.Series):
        super(Column, self).__subset__(instance, value)
        owner = self.__owner__
        key = self.__key__
        result: Column = owner.__dict__[key]
        if isinstance(owner, DataFrame):
            owner[key] = result
        assert self.__key__ in owner.attrs
        # noinspection PyUnresolvedReferences
        if (result.index != owner.index).any():
            raise ValueError(
                f"Index of column {self.__trace__} does not match owner index."
            )
        return result

    @truthy
    # def __align__(self, other: NDFrame = None):
    def __subalign__(self, owner: Frame):
        """
        align such that self.index == owner.index
        much like actual columns within a dataframe
        """

        o = owner.index
        s = self.index
        # noinspection PyTypeChecker,PyUnresolvedReferences

        if (
                len(o) == len(s)
                and (o == s).all()
        ):
            # no aligning needed
            return self

        if not o.difference(s).empty:
            raise ValueError(f"""
            Index of column {self.__trace__} is incompatible with index of the owner;
            be sure the Series returned by {self.__name__} has the same index as the owner.
            """)

        result = self.loc[o]
        # other.attrs[self.__key__] = result
        return result

    # todo: look deeper into overriding ior and iand
    def __ior__(self, other):
        print('ior')
        result = copy.copy(self)
        result = Series.__ior__(result, other)
        return result

    def __iand__(self, other):
        print('iand')
        result = copy.copy(self)
        result = Series.__iand__(result, other)
        return result



