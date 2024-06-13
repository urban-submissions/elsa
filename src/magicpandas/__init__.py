from typing import Union, Callable
from typing import TypeVar
from typing import Any
T = TypeVar('T')
import pandas as pd

from magicpandas.logger.logger import logger
from magicpandas.magic.imports import imports
from magicpandas.magic.log import log
from magicpandas.magic.magic import Magic
from magicpandas.magic.util import LazyModuleLoader
from magicpandas.pandas.cached import cached
from magicpandas.pandas.column import Column
from magicpandas.magic.default import default
from magicpandas.magic import util
from magicpandas.pandas.frame import Frame
from magicpandas.pandas.series import Series
from magicpandas.pandas.index import index

series = Series

# column = Union[Series, pd.Index]
# column = pd.Series

# def colfunc(func: T) -> Series | pd.Series | pd.Index | T:
#     return func
# column = Union[colfunc, Column]
# column = Union[Column, pd.Series]
# column: Column | Series
# column: pd.Series
# column = colfunc

def column(func: T) -> Union[pd.Series, Series, Column, Magic, T]:
    return func

globals()['column'] = Column

frame = Frame


if False:
    from magicpandas.pandas import geo
locals()['geo'] = LazyModuleLoader('magicpandas.pandas.geo')

if False:
    import magicpandas.graph as graph
locals()['graph'] = LazyModuleLoader('magicpandas.graph')

if False:
    import magicpandas.raster as raster
locals()['raster'] = LazyModuleLoader('magicpandas.raster')

from magicpandas.magic import globals

def __getitem__(*args, **kwargs):
    # allows def func() -> magic[int]
    ...

