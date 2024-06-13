from __future__ import annotations

from functools import *
from types import *
from typing import *
from typing import Self

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoSeries
from pandas import Series

import magicpandas as magic
# from elsa.resource import Resource
from elsa import resource

if False:
    import elsa.combos

E = RecursionError, AttributeError


def norm(func):
    """Assure that the result is correctly normalized."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result: np.ndarray = func(*args, **kwargs)

        # round to 8 decimal places
        tolerance = 1e-8
        loc = (np.abs(result - 1) < tolerance) | (np.abs(result) < tolerance)
        result = np.where(loc, np.round(result), result)

        loc = (result >= -tolerance) & (result <= 1.)
        assert loc.all(), 'result is not correctly normalized'

        return result

    return wrapper


def positive(func):
    """Assure that the result is positive."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result: np.ndarray = func(*args, **kwargs)

        tolerance = -1e-8
        loc = result > tolerance
        loc &= result < 0
        result = np.where(loc, 0., result)

        loc = result >= 0
        assert loc.all(), 'result is not positive'
        return result

    return wrapper


class Base(
    resource.Resource,
    magic.geo.Frame,
):
    """
    Base class which contains dynamic columns relating to the spatial
    qualities of subclasses. For a given method, it tries to calculcate
    the result from whichever columns are available.
    """
    loc: MappingProxyType[Any, Self] | Self
    iloc: MappingProxyType[Any, Self] | Self

    @magic.column.from_options(no_recursion=True)
    @positive
    def w(self) -> Series[float]:
        try:
            return self.x.values - self.width.values / 2
        except E:
            ...
        try:
            return self.image_width * self.normw.values

        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def s(self) -> Series[float]:
        try:
            return self.y.values - self.height.values / 2
        except E:
            ...
        try:
            return self.image_height * self.norms.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def e(self) -> Series[float]:
        try:
            return self.x.values + self.width.values / 2
        except E:
            ...
        try:
            return self.image_width * self.norme.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def n(self) -> Series[float]:
        try:
            return self.y.values + self.height.values / 2
        except E:
            ...
        try:
            return self.image_height * self.normn.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def x(self) -> Series[float]:
        try:
            return (self.w.values + self.e.values) / 2
        except E:
            ...
        try:
            return self.image_width * self.normx.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def y(self) -> Series[float]:
        try:
            return (self.s.values + self.n.values) / 2
        except E:
            ...
        try:
            return self.image_height.values * self.normy.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def height(self) -> Series[float]:
        try:
            return self.n.values - self.s.values
        except E:
            ...
        try:
            return self.image_height * self.normheight
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def width(self) -> Series[float]:
        try:
            return self.e.values - self.w.values
        except E:
            ...
        try:
            return self.image_width * self.normwidth
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normw(self) -> Series[float]:
        try:
            return self.w.values / self.image_width.values
        except E:
            ...
        try:
            return self.normx.values - self.normwidth.values / 2
        except E:
            ...
        try:
            return self.norme.values - self.normwidth.values
        except E:
            ...
        try:
            return self.fw.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def norms(self) -> Series[float]:
        try:
            return self.s.values / self.image_height.values
        except E:
            ...
        try:
            return self.normy.values - self.normheight.values / 2
        except E:
            ...
        try:
            return self.normn.values - self.normheight.values
        except E:
            ...
        try:
            return self.fs.values - self.nfile
        except E:
            ...

        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def norme(self) -> Series[float]:
        try:
            return self.e.values / self.image_width.values
        except E:
            ...
        try:
            return self.normx.values + self.normwidth.values / 2
        except E:
            ...
        try:
            return self.normw.values + self.normwidth.values
        except E:
            ...
        try:
            return self.fe.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normn(self) -> Series[float]:
        try:
            return self.n.values / self.image_height.values
        except E:
            ...
        try:
            return self.normy.values + self.normheight.values / 2
        except E:
            ...
        try:
            return self.norms.values + self.normheight.values
        except E:
            ...
        try:
            return self.fn.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normx(self) -> Series[float]:
        try:
            return self.x.values / self.image_width.values
        except E:
            ...
        try:
            return (self.normw.values + self.norme.values) / 2
        except E:
            ...
        try:
            return self.fw.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normy(self) -> Series[float]:
        try:
            return self.y.values / self.image_height.values
        except E:
            ...
        try:
            return (self.norms.values + self.normn.values) / 2
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normwidth(self) -> Series[float]:
        try:
            return self.width.values / self.image_width.values
        except E:
            ...
        try:
            return self.norme.values - self.normw.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @norm
    @positive
    def normheight(self) -> Series[float]:
        try:
            return self.height.values / self.image_height.values
        except E:
            ...
        try:
            return self.normn.values - self.norms.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    @positive
    def fn(self) -> Series[float]:
        return self.normn + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fs(self) -> Series[float]:
        return self.norms + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fe(self) -> Series[float]:
        return self.norme + self.nfile

    @magic.column.from_options(no_recursion=True)
    @positive
    def fw(self) -> Series[float]:
        return self.normw + self.nfile

    @magic.column
    def area(self) -> magic[float]:
        result = self.width.values * self.height.values
        return result

    @property
    def xmin(self):
        return self.w

    @xmin.setter
    def xmin(self, value):
        self.w = value

    @xmin.deleter
    def xmin(self):
        del self.w

    @property
    def ymin(self):
        return self.s

    @ymin.setter
    def ymin(self, value):
        self.s = value

    @ymin.deleter
    def ymin(self):
        del self.s

    @property
    def xmax(self):
        return self.e

    @xmax.setter
    def xmax(self, value):
        self.e = value

    @xmax.deleter
    def xmax(self):
        del self.e

    @property
    def ymax(self):
        return self.n

    @ymax.setter
    def ymax(self, value):
        self.n = value

    @ymax.deleter
    def ymax(self):
        del self.n


    @magic.column
    def image_height(self) -> Series[float]:
        return self.images.height.loc[self.ifile].values

    @magic.column
    def image_width(self) -> Series[float]:
        return self.images.width.loc[self.ifile].values

    @magic.geo.column
    def geometry(self):
        return shapely.box(self.fw, self.fs, self.fe, self.fn)


Boxes = Base
