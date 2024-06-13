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
# import elsa.combos as combos
import elsa.combos.combos as combos

if False:
    from .truth import Truth


class Combos(combos.Combos):
    __outer__: Truth

    @magic.column
    def ilabels(self) -> magic[tuple[int]]:
        """Returns a tuple of label IDs for each box; used for debugging"""
        ann = self.__outer__
        _ = ann.ibox, ann.ilabel
        result = (
            ann
            .reset_index()
            .drop_duplicates('ibox ilabel'.split())
            .sort_values('ilabel')
            .groupby('ibox', sort=False)
            .ilabel
            .apply(tuple)
            .loc[self.ibox]
            .values
        )
        return result
