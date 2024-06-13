from __future__ import annotations

import functools

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
from elsa.logits.alg1 import Categories, algorithm_1


if False:
    from .logits import Logits

class Unified(Logits):
    def __from_inner__(self) -> Self:
        ...
