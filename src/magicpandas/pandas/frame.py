from __future__ import annotations

import pandas as pd
from typing import *

import pandas

from magicpandas.magic.classboundstrings import ClassBoundStrings
from magicpandas.magic.order import Order
from magicpandas.magic.truthy import truthy
from magicpandas.pandas.cached import cached
from magicpandas.pandas.ndframe import NDFrame


class Frame(
    NDFrame,
    pandas.DataFrame,
):
    __init_nofunc__ = pandas.DataFrame.__init__
    __flush__ = ClassBoundStrings()
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]

    if False:
        def __call__(
                self,
                data=None,
                index: Axes | None = None,
                columns: Axes | None = None,
                dtype: Dtype | None = None,
                copy: bool | None = None,
        ) -> Self:
            ...

    def __fspath__(self):
        # cwd/magic/magic.pkl
        return self.__directory__ + '.parquet'

    @truthy
    def __from_file__(self):
        return pandas.read_parquet(self)

    def __to_file__(self, value=None):
        if value is None:
            value = self
        future = self.__root__.__threads__.submit(self.to_parquet, value)
        self.__root__.__futures__.append(future)

    @cached.property
    def __permanent__(self) -> set[str]:
        return set()

    @cached.property
    def __columns__(self) -> set[str]:
        return set()

    def __flush_columns__(self, columns: str | list[str] = None) -> Self:
        result = self.copy()
        permanent = self.__permanent__
        __columns__ = self.__columns__
        if columns is None:
            columns = self.columns
        elif isinstance(columns, str):
            columns = {columns}
        else:
            columns: set[str] = set(columns)
        # todo: only delete ones that are magic columns

        for column in columns:
            # todo: x permanent is not properly propagating
            try:
                self.__getnested__(column)
            except AttributeError:
                continue
            if (
                    column in __columns__
                    and column not in permanent
            ):
                del result[column]

        return result

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
                result += f'{pandas.DataFrame.__repr__(self)}'

        return result

    if False:
        def assign(self, **kwargs) -> Self:
            ...

    def __eq__(self, other):
        # temporary solution to an annoying problem:
        # pd.concat wants to do this:
        # check_attrs = all(objs.attrs == attrs for objs in other.objs[1:])
        # when we have magic frames cached in the attrs, even if weakrefed
        # as of python 3.10, they are compared, and this raises an exception
        # todo: add this to base magicpandas
        if isinstance(other, (pd.DataFrame, pd.Series)):
            return False
        return super().__eq__(other)

Frame.__call__