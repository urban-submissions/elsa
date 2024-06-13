from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import Series
from typing import Self

import magicpandas as magic
from elsa.annotation.invalid import Invalid
from elsa.annotation.rephrase import Rephrase
from elsa.resource import Resource

if False:
    from .annotation import Annotation


class Unique(Resource):
    __outer__: Annotation
    rephrase = Rephrase()
    invalid = Invalid()

    def __from_inner__(self) -> Self:
        """
        Called when accessing Annotation.unique to instantiate Unique
        """
        truth = self.__outer__
        names = 'isyns isyn'.split()
        _ = truth.isyns
        result = (
            truth
            .reset_index()
            .drop_duplicates(names)
            .set_index('isyns')
        )
        return result

    consumed: Self
    def consumed(self):
        """The subset of Unique after consuming the labels."""
        names = 'isyns isyn'.split()
        arrays = self.isyns, self.isyn.values
        needles = pd.MultiIndex.from_arrays(arrays, names=names)

        consumed = self.rephrase.consumed
        arrays = consumed.isyns, consumed.isyn.values
        haystack = pd.MultiIndex.from_arrays(arrays, names=names)

        loc = needles.isin(haystack)
        result = self.loc[loc]

        assert result.isyns.isin(self.isyns).all()
        return result

    @magic.column
    def isyn(self) -> magic[int]:
        """an isyn that belongs to the unique combination"""

    @magic.index
    def isyns(self) -> magic[tuple[int]]:
        """all isyns that belong to the unique combination"""

    @magic.column
    def meta(self):
        result = (
            self.synonyms
            .drop_duplicates('isyn')
            .set_index('isyn')
            .meta
            .loc[self.isyn]
            .values
        )
        return result

    @magic.cached.property
    def tuples(self) -> magic[tuple[int]]:
        result = (
            self
            .reset_index()
            .sort_values('isyn')
            .groupby('isyns', sort=False)
            .isyn
            .apply(tuple)
        )
        return result

    def indexed(self) -> Self:
        result = self.set_index('isyns')
        return result

    def includes(
            self,
            label: str = None,
            meta: str = None,
    ) -> Series[bool]:
        if label and meta:
            raise ValueError('label and meta cannot both be provided')
        if label is not None:
            if isinstance(label, str):
                isyn = self.synonyms.label2isyn[label]
            elif isinstance(label, int):
                isyn = label
            else:
                raise TypeError(f'label must be str or int, not {type(label)}')
            loc = self.isyn == isyn
        elif meta is not None:
            loc = self.meta == meta
        else:
            raise ValueError('label or meta must be provided')
        result = (
            Series(loc)
            .groupby(self.index.names)
            .any()
        )
        return result

    def excludes(
            self,
            label: str = None,
            meta: str = None,
    ) -> Series[bool]:
        return ~self.includes(label, meta)

    def synonymous(self, label: str) -> Series[bool]:
        isyn = self.synonyms.isyn.loc[label]
        loc = self.isyn == isyn
        return loc

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ibox and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        ann = self.__outer__
        result = Series(0, index=self.isyns)
        isyns = self.isyns[loc]
        update = (
            ann.isyn
            .loc[loc]
            .groupby(isyns)
            .nunique()
        )
        result.update(update)
        result = result.set_axis(self.index)
        return result

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ibox and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        result = (
            self.isyn
            .loc[loc]
            .groupby(level='isyns')
            .nunique()
        )
        return result

    @magic.column
    def cardinal(self):
        result = (
            self.elsa.synonyms
            .drop_duplicates('isyn')
            .reset_index()
            .set_index('isyn')
            .syn.loc[self.isyn]
            .values
        )
        return result

    @magic.series
    def alone_appendix(self) -> magic[str]:
        """If only a singular subject is present, returns ' alone'"""

        # Later: we talked about only having "alone" with person! not with an individual.
        includes = self.includes
        excludes = self.excludes
        loc = (
                includes('alone')
                ^ includes('laborer')
                ^ includes('vendor')
                ^ includes('kid')
                ^ includes('teenager')
                ^ includes('elderly')
                ^ includes('baby')
        )
        loc &= excludes('couple')
        loc &= excludes('group')
        loc: Series[bool]
        data = np.where(loc, ' alone', '')
        result = Series(data, index=loc.index)
        return result
