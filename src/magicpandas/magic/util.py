from __future__ import annotations
import ast
import inspect
import textwrap
from collections import Counter
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
from typing import Self
from dataclasses import dataclass
import numpy as np
from numpy import ndarray


import ast
import inspect
import textwrap


def has_uncommented_return(func):
    source_code = inspect.getsource(func)
    source_code = textwrap.dedent(source_code)
    tree = ast.parse(source_code)

    class ReturnVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Return(self, node):
            if (
                    not isinstance(node.value, ast.Constant)
                    or node.value.value is not None
            ):
                self.found = True

    visitor = ReturnVisitor()
    visitor.visit(tree)
    return visitor.found

class LazyModuleLoader:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load_module(self):
        if self.module is None:
            self.module = __import__(self.module_name, fromlist=[''])

    def __getattr__(self, item):
        self._load_module()
        return getattr(self.module, item)

    def __setattr__(self, key, value):
        if key in ('module_name', 'module'):
            super().__setattr__(key, value)
        else:
            self._load_module()
            setattr(self.module, key, value)


def noreturn(func) -> bool:
    tree = ast.parse(inspect.getsource(func))
    for node in ast.walk(tree):
        if isinstance(node, ast.Return):
            return True
    return False

class ReturnVisitor(ast.NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Return(self, node):
        self.found = True
        self.generic_visit(node)

def returns(code):
    if inspect.isfunction(code):
        code = inspect.getsource(code)
    dedented_code = textwrap.dedent(code)
    tree = ast.parse(dedented_code)
    visitor = ReturnVisitor()
    visitor.visit(tree)
    return visitor.found



def has_executable_code(func):
    tree = ast.parse(inspect.getsource(func))
    for node in ast.walk(tree):
        clses = (
            ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For, ast.While,
            ast.If, ast.With, ast.Call, ast.Expr, ast.AsyncFor,
            ast.AsyncWith, ast.Try, ast.ExceptHandler, ast.FunctionDef, ast.ClassDef,
        )
        if isinstance(node, clses):
            return True
    return False

@dataclass
class Constituents:
    unique: ndarray
    ifirst: ndarray
    ilast: ndarray
    istop: ndarray
    repeat: ndarray

    @cached_property
    def indices(self) -> ndarray:
        return np.arange(len(self)).repeat(self.repeat)

    def __len__(self):
        return len(self.unique)

    def __repr__(self):
        return f'Constituents({self.unique}) at {hex(id(self))}'

    def __getitem__(self, item) -> Constituents:
        unique = self.unique[item]
        ifirst = self.ifirst[item]
        ilast = self.ilast[item]
        istop = self.istop[item]
        repeat = self.repeat[item]
        con = Constituents(unique, ifirst, ilast, istop, repeat)
        return con


def constituents(self: Union[Series, ndarray, Index], monotonic=True) -> Constituents:
    try:
        monotonic = self.is_monotonic_increasing
    except AttributeError:
        pass
    if monotonic:
        if isinstance(self, (Series, Index)):
            assert self.is_monotonic_increasing
        elif isinstance(self, ndarray):
            assert np.all(np.diff(self) >= 0)

        unique, ifirst, repeat = np.unique(self, return_counts=True, return_index=True)
        istop = ifirst + repeat
        ilast = istop - 1
        # constituents = Constituents(unique, ifirst, ilast, istop, repeat)
        constituents = Constituents(
            unique=unique,
            ifirst=ifirst,
            ilast=ilast,
            istop=istop,
            repeat=repeat,
        )
    else:
        counter = Counter(self)
        count = len(counter)
        repeat = np.fromiter(counter.values(), dtype=int, count=count)
        unique = np.fromiter(counter.keys(), dtype=self.dtype, count=count)
        val_ifirst: dict[int, int] = dict()
        val_ilast: dict[int, int] = {}
        for i, value in enumerate(self):
            if value not in val_ifirst:
                val_ifirst[value] = i
            val_ilast[value] = i
        ifirst = np.fromiter(val_ifirst.values(), dtype=int, count=count)
        ilast = np.fromiter(val_ilast.values(), dtype=int, count=count)
        istop = ilast + 1
        constituents = Constituents(unique, ifirst, ilast, istop, repeat)

    return constituents


class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_code = False

    def visit_Assign(self, node):
        self.has_code = True
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.has_code = True
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self.has_code = True
        self.generic_visit(node)

    def visit_Return(self, node):
        self.has_code = True
        self.generic_visit(node)

def contains_functioning_code(code):
    if inspect.isfunction(code):
        code = inspect.getsource(code)
    dedented_code = textwrap.dedent(code)
    tree = ast.parse(dedented_code)
    visitor = CodeVisitor()
    visitor.visit(tree)
    return visitor.has_code


if __name__ == '__main__':
    def false():
        """hello world"""


    def true():
        """hello world"""
        x = 1 + 2

    def also_true():
        return 1

    assert contains_functioning_code(true)
    assert not contains_functioning_code(false)
    assert contains_functioning_code(also_true)


