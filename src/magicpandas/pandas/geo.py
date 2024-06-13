from __future__ import annotations

import pathlib
import re
from functools import singledispatch
from typing import *

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from geopandas import GeoDataFrame
from geopandas.array import GeometryDtype
from geopandas.base import is_geometry_type
from geopandas.geoseries import GeoSeries
from pandas.core.internals import SingleBlockManager
from shapely.geometry import Point
from shapely.geometry import box
from shapely.ops import transform

import magicpandas as magic
from magicpandas.pandas.cached import cached
from typing import TypeVar
T = TypeVar('T')

# from magicpandas.pandas.series import Series
# from magicpandas.pandas.frame import Frame
# import magicpandas as magic
# from magicpandas.pandas import

__all__ = ['Frame', 'Series']

if False:
    from .ndframe import NDFrame


class GeoX:
    """
    Allows the user to more conveniently perform a spatial query;
    GDF.cx[...] has the requirements that the coordinates match
    the GDF crs, and that the coordinates are in the order (x, y).
    This method allows the user to use (y, x) coordinates, and
    automatically transforms from 4326 to the GDF crs.
    """
    _regex = re.compile(r'(-?\d+\.?\d*),(-?\d+\.?\d*)')

    # todo: prevent memory leak; just return a copy with self.__self__ = instance
    def __get__(self, instance: NDFrame, owner):
        self.instance = instance
        self.owner = owner
        return self

    def __getitem__(self, item):
        """
        artifacts.geox['y,x':'y,x']
        artifacts.geox[(y,x):(y,x)]
        artifacts.geox[y,x:y,x]
        artifacts.geox[y,x,y,x]
        :return: Source
        """
        instance = self.instance

        if isinstance(item, slice):
            start = item.start
            stop = item.stop
            if isinstance(start, str):
                # todo
                raise NotImplementedError('my regex is apparently wrong')
                match = self._regex._match(start)
                s = float(match.subnode(1))
                w = float(match.subnode(2))
            else:
                s, w = start
            if isinstance(stop, str):
                match = _regex._match(stop)
                n = float(match.subnode(1))
                e = float(match.subnode(2))
            else:
                n, e = stop

        elif (
                isinstance(item, tuple)
                and len(item) == 3
                and isinstance(item[1], slice)
        ):
            s = item[0]
            w = item[1].start
            n = item[1].stop
            e = item[2]

        elif (
                isinstance(item, tuple)
                and len(item) == 4
        ):
            s, w, n, e = item

        else:
            raise ValueError('invalid refx slice: %s' % item)

        w, e = min(w, e), max(w, e)
        s, n = min(s, n), max(s, n)
        trans = pyproj.Transformer.from_crs(4326, instance.crs, always_xy=True).transform
        w, s = trans(w, s)
        e, n = trans(e, n)
        result = instance.cx[w:e, s:n]
        result = instance(result)
        instance.__call__
        return result


T = TypeVar('T')


class Indexable(Protocol):
    def __getitem__(self, key) -> T:
        ...


class Frame(
    magic.Frame,
    gpd.GeoDataFrame,
):
    geox: Indexable[Frame] = GeoX()
    __init_nofunc__ = gpd.GeoDataFrame.__init__
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]

    if False:
        def __call__(self, data=None, *args, geometry=None, crs=None, **kwargs):
            ...

    @cached.property
    def __bounds__(self) -> tuple[float]:
        bounds = self.total_bounds
        current_crs = self.crs
        target_crs = 'EPSG:4326'

        if current_crs != target_crs:
            project = pyproj.Transformer.from_crs(current_crs, target_crs, always_xy=True).transform
            bounds_4326 = transform(project, box(*bounds)).bounds
        else:
            bounds_4326 = bounds

        return bounds_4326

    @cached.root.property
    def __rootdir__(self) -> str:
        result = super().__rootdir__
        result = result + '_'.join(
            round(x, 2).__str__()
            for x in self.__bounds__
        )
        return result

    def explore(
            self,
            *args,
            tiles='cartodbdark_matter',
            **kwargs
    ) -> folium.Map:
        import folium
        m = explore(self, color='grey', geometry='geometry', *args, **kwargs, tiles=tiles)
        folium.LayerControl().add_to(m)
        return m

    if False:
        explore = GeoDataFrame.explore

    # always construct the class; do not construct pd.DataFrames
    def _constructor_from_mgr(self, mgr, axes):
        return self.__class__._from_mgr(mgr, axes)

    @property
    def _constructor_sliced(self):
        def _geodataframe_constructor_sliced(*args, **kwargs):
            """
            A specialized (Geo)Series constructor which can fall back to a
            Series if a certain operation does not produce geometries:

            - We only return a GeoSeries if the data is actually of geometry
              dtype (and so we don't try to convert geometry objects such as
              the normal GeoSeries(..) constructor does with `_ensure_geometry`).
            - When we get here from obtaining a row or column from a
              GeoDataFrame, the goal is to only return a GeoSeries for a
              geometry column, and not return a GeoSeries for a row that happened
              to come from a DataFrame with only geometry dtype columns (and
              thus could have a geometry dtype). Therefore, we don't return a
              GeoSeries if we are sure we are in a row selection case (by
              checking the identity of the index)
            """
            srs = pd.Series(*args, **kwargs)
            is_row_proxy = srs.index is self.columns
            if is_geometry_type(srs) and not is_row_proxy:
                srs = GeoSeries(srs)
            return srs

        return _geodataframe_constructor_sliced

    def trans_xy_to(
            self,
            crs
    ):
        result = (
            pyproj.Transformer
            .from_crs(self.crs, crs, always_xy=True)
            .transform
        )
        return result

    def trans_xy_from(
            self,
            crs
    ):
        result = (
            pyproj.Transformer
            .from_crs(crs, self.crs, always_xy=True)
            .transform
        )
        return result

    def __setter__(self, value):
        if isinstance(value, (str, pathlib.Path)):
            ext = pathlib.Path(value).suffix
            match ext:
                case '.feather':
                    self.__class__.read_feather(value)
                case '.parquet':
                    self.__class__.read_parquet(value)
                case _:
                    self.__class__.read_csv(value)
        return value


class Series(
    # magic.Column,
    magic.Series,
    gpd.GeoSeries,
    pd.Series,
):
    geox: Indexable[Series] = GeoX()
    __subinit__ = gpd.GeoSeries.__init__
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]

    if False:
        @staticmethod
        def __call__(self, data=None, index=None, crs: Optional[Any] = None, **kwargs) -> Self:
            ...

    @classmethod
    def _constructor(cls, data=None, index=None, crs: Optional[Any] = None, **kwargs):
        """
        A flexible constructor for GeoSeries._constructor, which needs to be able
        to fall back to a Series (if a certain operation does not produce
        geometries)
        """
        try:
            return cls(data=data, index=index, crs=crs, **kwargs)
        except TypeError:
            return pd.Series(data=data, index=index, **kwargs)

    def _constructor_from_mgr(self, mgr, axes):
        assert isinstance(mgr, SingleBlockManager)

        if not isinstance(mgr.blocks[0].dtype, GeometryDtype):
            return Series._from_mgr(mgr, axes)

        # return self.__class__._from_mgr(self, mgr, axes)
        return self._from_mgr(mgr, axes)


@singledispatch
def explore(
        self: Series | Frame,
        name=None,
        geometry='geometry',
        *args,
        **kwargs
) -> folium.Map:
    """Convenience wrapper to GeoDataFrame.explore"""
    self = self.copy()
    _ = self.geometry
    if isinstance(self, Series):
        # noinspection PyTypeChecker
        # self = self.reset_index()
        self = (
            self
            .reset_index()
            .rename(columns={self.__key__: 'geometry'})
        )
    kwargs['tiles'] = kwargs.setdefault('tiles', 'cartodbdark_matter')
    style_kwargs = kwargs.setdefault('style_kwds', {})
    style_kwargs.setdefault('weight', 5)
    style_kwargs.setdefault('radius', 5)

    if name is None:
        name = geometry
    if geometry not in self.columns:
        return kwargs['m']
    loc = self.dtypes != 'geometry'
    loc &= self.dtypes != 'object'
    loc |= self.columns == geometry
    # todo: better way other than checking first row?
    is_string = np.fromiter((
        isinstance(x, str)
        for x in self.iloc[0]
    ), dtype=bool, count=len(self.columns))
    loc |= is_string

    columns = self.columns[loc]
    loc = self[geometry].notna().values
    # folium = import_folium()
    # max_zoom=kwargs.get('max_zoom', 28)
    max_zoom = kwargs.setdefault('max_zoom', 28)
    zoom_start = kwargs.setdefault('zoom_start', 14)
    if not loc.any():
        try:
            return kwargs['m']
        except KeyError:
            result = folium.Map(name=name, max_zoom=max_zoom, zoom_start=zoom_start)
        return result
    self: GeoDataFrame = (
        self.loc[loc, columns]
        .reset_index()  # reset index so it shows in folium
        .set_geometry(geometry)
        .pipe(GeoDataFrame)
    )
    # if isinstance(self.geometry.iloc[0], MultiPoint):
    #     # LayerControl doesn't work with MultiPoints
    #     self = self.explode(column=geometry, ignore_index=True)

    if 'MultiPoint' in self.geom_type.unique():
        # LayerControl doesn't work with MultiPoints
        self = self.explode(column=geometry, ignore_index=True)

    m = GeoDataFrame.explore(
        self,
        *args,
        **kwargs,
        name=name,
    )
    return m


# todo column complains if not a GeoSeries passed
class Column(
    magic.Column,
    Series,
    pd.Series,
):
    __init_nofunc__ = gpd.GeoSeries.__init__

    # def __postprocess__(self, result: T) -> T:
    #     if not isinstance(result, gpd.GeoSeries):
    #         raise TypeError(
    #             f'{self.__class__.__name__} {self.__trace__} must be a GeoSeries to specify CRS; '
    #             f'got {type(result)}'
    #         )
    #     if not result.crs:
    #         warnings.warn(f'No CRS specified for {self.__trace__} ')
    #     return super().__postprocess__(result)

    def __subset__(self, instance, value):
        owner = self.__owner__
        key = self.__key__
        if (
                key == 'geometry'
                and isinstance(owner, gpd.GeoDataFrame)
        ):
            # todo: better way?
            owner._item_cache[key] = None
            owner.set_geometry(value, inplace=True)
            try:
                del owner._item_cache[key]
            except KeyError:
                ...
            return owner[key]
        else:
            return super().__subset__(instance, value)

    # def __set__(self, instance, value):
    #     # Series.__setold__(self, instance, value)
    #     Series.__subset__(self, instance, value)
    #     # super().__setold__(instance, value)
    #     owner = self.__owner__
    #     key = self.__key__
    #     result: Column = owner.__dict__[key]
    #     if isinstance(owner, gpd.GeoDataFrame):
    #         try:
    #             owner.set_geometry(result, inplace=True)
    #         except ValueError:
    #             # If you do something like sub=frame.iloc[...], frame.geometry,
    #             # the .attrs and .dict will not have updated, and it might be that the
    #             # original frame had duplicate indicies, so the sub geocolumn cannot be resolved.
    #             # here, we just assign the geometry to the native geopandas column, which
    #             # preserved the index.
    #             result = owner.geometry = owner['geometry']
    #
    #     elif isinstance(owner, pd.DataFrame):
    #         owner[key] = result
    #     # noinspection PyUnresolvedReferences
    #     if (result.index != owner.index).any():
    #         raise ValueError(
    #             f"Index of column {self.__trace__} does not match owner index."
    #         )
    #
    #     # assert self.attrs
    #     try:
    #         assert isinstance(owner.attrs[self.__key__](), self.__class__)
    #     except AssertionError:
    #         # Series.__setold__(self, instance, value)
    #         Series.__subset__(self, instance, value)
    #     assert isinstance(owner.__dict__[self.__key__](), self.__class__)


assert Column.__init__ is not pd.Series.__init__
column = Column
series = Series
frame = Frame


def import_folium() -> folium:
    # When debugging, importing folium seems to cause AttributeError: module 'posixpath' has no attribute 'sep'
    import posixpath
    sep = posixpath.sep
    import folium
    posixpath.sep = sep
    return folium


if __name__ == '__main__':
    # column of some random points
    column = Column([
        Point(1, 1),
        Point(2, 2),
        Point(3, 3),
    ])
    column.loc[:]
    assert isinstance(column.loc[:], Column)
# Column(([MultiPoint([(-1, -1), (1, 1)])]))
