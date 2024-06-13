from __future__ import annotations
import weakref
from pandas import DataFrame, Series, Index, MultiIndex

import pandas as pd

if False:
    from .ndframe import NDFrame

import functools


def update_attr_index(func):
    @functools.wraps(func)
    def wrapper(self: NDFrame, *args, **kwargs):
        cls = self.__class__
        result: NDFrame | pd.DataFrame | pd.Series = func(self, *args, **kwargs)
        # result.set_index()

        for key, value in result.attrs.items():
            if not (
                    isinstance(value, NDFrame)
                    and value.__align__
            ):
                continue
            value = value.set_axis(result.index, axis=0)
            # result.attrs[key] = result.__dict__[key] = value.set_index(result.index)

        return result

    return wrapper


