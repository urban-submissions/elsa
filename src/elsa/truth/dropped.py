from __future__ import annotations

from typing import *

import numpy as np
import pandas as pd
from pandas import Series

import magicpandas as magic
from elsa.truth.truth import Truth


def nboxes(truth: Truth) -> Series[int]:
    _ = truth.w, truth.s, truth.e, truth.n, truth.label, truth.file
    needles = truth['file w s e n'.split()].pipe(pd.MultiIndex.from_frame)
    haystack = needles.unique()
    ibox = pd.Series(np.arange(len(haystack)), index=haystack)
    result = (
        ibox
        .loc[needles]
        .groupby(truth.file.values)
        .nunique()
    )
    return result


class Dropped(Truth):
    __outer__: Truth

    @magic.series
    def internally(self) -> magic[int]:
        """How many truth boxes were dropped through internal processes"""
        result = self.total - self.externally
        return result

    @magic.series
    def externally(self) -> magic[int]:
        """How many truth boxes were dropped before it was passed to the constructor"""
        truth = self.__outer__
        original = truth.original
        truth = (
            truth
            .from_inferred(truth.passed)
            .pipe(truth)
        )
        original = nboxes(original)
        truth = nboxes(truth)
        result = (
            truth
            .reindex(original.index, fill_value=0)
            .pipe(original.sub)
        )
        return result

    @magic.series
    def total(self) -> magic[int]:
        """How many truth boxes were dropped either way"""
        truth = self.__outer__
        original = truth.original
        truth = nboxes(truth)
        original = nboxes(original)
        result = (
            truth
            .reindex(original.index, fill_value=0)
            .pipe(original.sub)
        )
        return result

    def __from_inner__(self) -> Self:
        """The subset of truth.original which was dropped"""
        t = self.__outer__
        o = t.original
        _ = o.label, o.w, o.s, o.e, o.n, o.file
        _ = t.label, t.w, t.s, t.e, t.n, t.file
        needles = o['label w s e n file'.split()].pipe(pd.MultiIndex.from_frame)
        haystack = t['label w s e n file'.split()].pipe(pd.MultiIndex.from_frame)
        loc = ~needles.isin(haystack)
        result = o.loc[loc]
        return result


