
from __future__ import annotations

from pandas import Series
from typing import Self

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.evaluation.grid import Grid

class Stacks(Resource):
    __outer__: Grid

    @magic.column
    def imatch(self):
        """truth box that the combo was matched to"""

    @magic.index
    def ibox(self):
        """truth box that the combo was matched to"""

    @magic.column
    def isyn(self):
        """synonym ID of the label """

    @magic.index
    def isyns(self):
        ...

    def __from_inner__(self) -> Self:
        grid = self.__outer__
        elsa = grid.elsa
        isyns = elsa.isyns
        size = isyns.groupby(level='isyns').size()
        repeat = size.loc[grid.isyns]
        imatch = grid.imatch.repeat(repeat).values
        ibox = grid.ibox.repeat(repeat).values
        isyn = isyns.isyn.loc[grid.isyns].values
        isyns = grid.isyns.repeat(repeat).values
        prompt = grid.prompt.repeat(repeat).values
        # duplicates are fine for includes, excludes, get_nunique
        result = (
            self(dict(
                imatch=imatch,
                ibox=ibox,
                isyns=isyns,
                isyn=isyn,
                prompt=prompt,
            ))
            .set_index('ibox isyns'.split())
        )
        return result

    @magic.column.from_options(dtype='category')
    def meta(self) -> Series[str]:
        """The metaclass, or type of label, for each box"""
        synonyms = self.synonyms
        _ = synonyms.meta
        result = (
            synonyms
            .drop_duplicates('isyn')
            .set_index('isyn')
            .meta
            .loc[self.isyn]
            .values
        )
        return result

    def includes(
            self,
            label: str | int = None,
            meta: str | int = None,
    ) -> Series[bool]:
        """Determine if a combo contains a label"""
        if label and meta:
            raise ValueError('label and meta cselfot both be provided')
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
            .groupby(self.ibox, sort=False)
            .any()
        )
        return result

    def excludes(
            self,
            label: str | int = None,
            meta: str | int = None,
    ) -> Series[bool]:
        """Determine if a combo excludes a label"""
        return ~self.includes(label, meta)

    def get_nunique(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the ibox and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        result = Series(0, index=self.ibox.unique())
        ibox = self.ibox[loc]
        update = (
            self.isyn
            .loc[loc]
            .groupby(ibox)
            .nunique()
        )
        result.update(update)
        return result

    @magic.column
    def truth_isyns(self) -> magic[tuple[int]]:
        result = (
            self.elsa.truth.isyns
            .loc[self.ibox]
            .values
        )
        return result


