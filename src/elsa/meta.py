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
from pathlib import Path

import itertools
from typing import *

import numpy as np
import pandas as pd

import magicpandas as magic

if False:
    from .root import Elsa


META = f"""
condition; state; activity; other
"""

class Meta(magic.Frame):
    __outer__: Elsa
    __owner__: Elsa

    @magic.index
    def imeta(self) -> Series[int]:
        """The unique index of the synonym set that the label belongs to"""

    @magic.column
    def meta(self) -> Series[str]:
        """The name of the meta category that describes labels"""

    def __from_inner__(self) -> Self:
        meta = META.split("; ")
        imeta = np.arange(len(meta))
        index = pd.Index(imeta, name='imeta')
        result = self({
            'meta': meta,
        }, index=index)
        return result

    @cached_property
    def meta2imeta(self) -> dict[str, int]:
        return Series(self.imeta, index=self.meta).to_dict()