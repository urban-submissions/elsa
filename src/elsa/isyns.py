from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import *
from pandas import Series, DataFrame

import magicpandas as magic

if False:
    from elsa import Elsa


class ISyns(magic.Frame):
    __outer__: Elsa

    def __from_inner__(self):
        """Called when accessing Elsa.isyns to instantiate ISyns"""
        elsa = self.__outer__
        isyns = elsa.truth.isyns.unique()
        repeat = np.fromiter(map(len, isyns), int, len(isyns))
        isyn = np.fromiter(chain.from_iterable(isyns), int, repeat.sum())
        isyns = isyns.repeat(repeat)
        index = pd.Index(isyns, name='isyns')
        result = self({
            'isyn': isyn
        }, index=index)
        return result

    @magic.column
    def isyn(self):
        """
        An ordered tuple of the synonym IDs representing a given combo.
        For example, "person walking" would have isyns=(0, 1), and
        "pedestrian strolling" would also have isyns=(0, 1), as they are
        synonymous.
        """

    @magic.index
    def isyns(self):
        """
        An ordered tuple of the synonym IDs representing a given combo.
        For example, "person walking" would have isyns=(0, 1), and
        "pedestrian strolling" would also have isyns=(0, 1), as they are
        synonymous.
        """


    @magic.Frame
    def subcombo(self) -> DataFrame:
        """
        starts with

        index: all unique isyns
        columns: all unique isyns
        value: True if index is a subset of columns

        needle  haystack    value
        0       0           True
        0       1           False
        0       2           False
        1       0           False

        ends with
                (0,)    (0,1),  (1,2)
        (0,)    1       1       0
        (0,1)   0       1       0
        (1,2)   0       0       1
        """

        isyns = self.isyns
        names = 'i c'.split()
        pairs = pd.MultiIndex.from_product((isyns, isyns), names=names)

        i = pairs.get_level_values('i')
        c = pairs.get_level_values('c')
        size = (
            self
            .groupby(level='isyns', sort=False)
            .size()
        )

        names = 'isyns isyn'.split()

        # haystack is just unique isyns
        haystack = (
            self
            .reset_index()
            [names]
            .pipe(pd.MultiIndex.from_frame)
        )

        # needles are cartesian pairs between haystack.isyns and isyns
        repeat = size.loc[i].values
        isyns = c.repeat(repeat)
        isyn = self.isyn.loc[i].values
        arrays = isyns, isyn
        needles = pd.MultiIndex.from_arrays(arrays, names=names)

        # each needle, haystack pair is repeated by haystack.isyns size
        names = 'r c'.split()
        loc = needles.isin(haystack)
        arrays = i.repeat(repeat), c.repeat(repeat)
        index = pd.MultiIndex.from_arrays(arrays, names=names)

        result: DataFrame = (
            Series(loc, index=index)
            .groupby(level=names)
            .all()
            .astype(bool)
            .unstack(level='c')
        )
        return result
