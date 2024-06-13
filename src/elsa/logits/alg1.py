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

import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from functools import *
from pandas import Index, MultiIndex
from pathlib import Path
from typing import *

import magicpandas as magic


from elsa.predictions import Predictions
if False:
    from elsa.logits.logits import Logits

# 9 prompts, 3 of each condition( group of people, ndividual and couple)
# for each prompt, show which image generated the highest summed confidence

class Categories(magic.Frame):
    __outer__: Logits
    @magic.cached.cmdline.property
    def low(self) -> float:
        return .3

    @magic.cached.cmdline.property
    def medium(self) -> float:
        return .6

    @magic.cached.cmdline.property
    def high(self):
        return .8

    def __from_inner__(self) -> Self:
        logits = self.__outer__
        with self.configure:
            inf = float('inf')
            bins = -inf, self.low, self.medium, self.high, inf
        labels = 'low medium high ultra'.split()
        func = partial(pd.cut, bins=bins, labels=labels)
        result = logits.maximums.labels.apply(func)

        return result

def algorithm_1(
        self: Logits,
) -> Predictions:
    # for
    loc = self.categories != 'low'
