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
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path

from functools import *
from pandas import Series

import magicpandas as magic

if False:
    from .scored import Scored


class AveragePrecision(magic.Frame):
    __outer__: Scored

    def __from_inner__(self) -> Self:
        scored = self.__outer__
        c = {
            key: value.tolist()
            for key, value in scored.c.ap_torch.items()
        }
        cs = {
            key: value.tolist()
            for key, value in scored.cs.ap_torch.items()
        }
        csa = {
            key: value.tolist()
            for key, value in scored.csa.ap_torch.items()
        }
        csao = {
            key: value.tolist()
            for key, value in scored.csa.ap_torch.items()
        }
        overall = {
            key: value.tolist()
            for key, value in scored.ap_torch.items()
        }
        person = {
            key: value.tolist()
            for key, value in scored.person.ap_torch.items()
        }
        pair = {
            key: value.tolist()
            for key, value in scored.pair.ap_torch.items()
        }
        people = {
            key: value.tolist()
            for key, value in scored.people.ap_torch.items()
        }
        c["level"] = "c"
        cs["level"] = "cs"
        csa["level"] = "csa"
        csao["level"] = "csao"
        overall["level"] = "overall"
        person["level"] = "person"
        pair["level"] = "pair"
        people["level"] = "people"
        data = c, cs, csa, csao, overall, person, pair, people
        result = (
            self(data)
            .set_index("level")
        )
        return result

    @cached_property
    def c(self) -> Series:
        return self.xs('c')

    @cached_property
    def cs(self) -> Series:
        return self.xs('cs')

    @cached_property
    def csa(self) -> Series:
        return self.xs('csa')

    @cached_property
    def csao(self) -> Series:
        return self.xs('csao')

    @magic.index
    def level(self):
        ...

    @magic.column
    def mar_small(self) -> magic[float]:
        ...

    @magic.column
    def mar_medium(self) -> magic[float]:
        ...

    @magic.column
    def mar_large(self) -> magic[float]:
        ...

    @magic.column
    def mar_per_class(self) -> magic[float]:
        ...


