from __future__ import annotations

import itertools
import numpy as np
from itertools import chain
from pandas import Series
from typing import Self

import magicpandas as magic
from elsa.annotation.prompts import Prompts
from elsa.resource import Resource

if False:
    import elsa.annotation.upgrade
    import elsa.annotation.consumed
    from .unique import Unique
    from elsa.annotation.consumed import Consumed


class Rephrase(Resource):
    """A synonymous combination of labels for every unique combination of isyn"""
    __outer__: Unique
    prompts = Prompts()

    def __from_inner__(self) -> Self:
        uniques = self.__outer__
        truth = uniques.__outer__
        elsa = truth.__outer__

        def cartesian(it_synonyms: Series):
            return list(itertools.product(*it_synonyms))

        try:
            synonyms = (
                elsa.synonyms.prompts.synonyms
                .groupby('isyn', sort=False)
                .apply(list)
                .loc[uniques.isyn]
                .groupby(uniques.isyns, sort=False)
                .apply(cartesian)
            )
        except KeyError as e:
            loc = ~uniques.isyn.isin(elsa.synonyms.prompts.isyn)
            isyn = uniques.isyn.loc[loc].unique()
            loc = elsa.synonyms.isyn.isin(isyn)
            synonyms = elsa.synonyms.loc[loc].syn
            raise KeyError(
                f'Some synonyms are likely missing from PROMPTS: {synonyms}'
            ) from e
        it = map(len, synonyms)
        first = np.fromiter(it, int, len(synonyms))
        it = map(len, chain.from_iterable(synonyms))
        count = first.sum()
        second = np.fromiter(it, int, count)
        isyns = (
            synonyms.index
            .repeat(first)
            .repeat(second)
        )
        irephrases = np.arange(first.sum()).repeat(second)
        it = chain.from_iterable(chain.from_iterable(synonyms))
        count = second.sum()
        label = np.fromiter(it, object, count)
        # result = (
        #     self({
        #         'isyns': isyns,
        #         'irephrase': irephrases,
        #         'label': label,
        #     })
        #     # .set_index(['isyns', 'irephrase'])
        # )
        result = self({
            'isyns': isyns,
            'irephrase': irephrases,
            'label': label,
        })
        _ = result.iorder
        result = (
            result
            .sort_values('isyns irephrase iorder'.split())
            .set_index('isyns irephrase'.split())
        )
        # _ = result.ilast
        return result

    consumed: elsa.annotation.consumed.Consumed

    def consumed(self) -> Consumed:
        """
        A subset of the Rephrase where:
            'consumer' labels have their natural label modified
            'consumed' labels are dropped
        """

    # def joined(self) -> elsa.annotation.upgrade.Joined:
    #     """Join the labels; used in prompt generation"""
    #
    # def reworded(self) -> elsa.annotation.upgrade.Reworded:
    #     """Reword the labels; used in prompt spans during prediction"""
    #
    @magic.index
    def isyns(self) -> magic[int]:
        """Identifier for each unique combination of isyn"""

    @magic.index
    def irephrase(self) -> magic[int]:
        """Identifier for each rephrase for each unique combination of isyn"""

    @magic.column
    def iorder(self) -> magic[int]:
        result = (
            self.elsa.synonyms.iorder
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def is_in_coco(self) -> magic[bool]:
        """False where not in labels metadata or not in is_coco"""
        synonyms = self.elsa.synonyms
        is_in_coco = (
            synonyms.is_in_coco
            .get(self.label, False)
            .values
        )
        return is_in_coco

    @magic.column
    def is_like_coco(self) -> magic[bool]:
        """False where not in labels metadata or not in is_coco"""
        synonyms = self.elsa.synonyms

        is_like_coco = (
            synonyms.is_like_coco
            .get(self.label, False)
            .values
        )
        return is_like_coco

    @magic.column
    def label(self) -> magic[str]:
        ...

    @magic.column
    def natural(self):
        result = (
            self.elsa.synonyms.prompts.natural
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def isyn(self):
        result = (
            self.elsa.synonyms.isyn
            .loc[self.label]
            .values
        )
        return result

    # @magic.column
    # def ifirst(self) -> magic[int]:
    #     prompts = (
    #         self.reworded.prompts.natural
    #         .reset_index('isyns')
    #         .natural
    #         .loc[self.irephrase]
    #     )
    #     labels = self.reworded.natural
    #     ifirst = np.fromiter((
    #         prompt.index(phrase)
    #         for prompt, phrase in zip(prompts, labels)
    #     ), dtype=int, count=len(self))
    #     return ifirst
    #
    # @magic.column
    # def ilast(self) -> magic[int]:
    #     result = (
    #             self.reworded.natural
    #             .str.len()
    #             .values
    #             + self.ifirst.values
    #     )
    #     return result
    #
    # @magic.column
    # def global_ilast(self):
    #     result = (
    #             self.prompts.global_ilast
    #             .reset_index()
    #             .set_index('irephrase')
    #             .global_ilast
    #             .loc[self.irephrase]
    #             .values
    #             + self.ilast.values
    #     )
    #     return result
    #
    # @magic.column
    # def global_ifirst(self):
    #     result = (
    #             self.prompts.global_ilast
    #             .reset_index()
    #             .set_index('irephrase')
    #             .global_ilast
    #             .loc[self.irephrase]
    #             .values
    #             + self.ilast.values
    #     )
    #     return result
    #

    @magic.column
    def meta(self):
        result = (
            self.synonyms.meta
            .loc[self.label]
            .values
        )
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
            .groupby(self.isyns, sort=False)
            .any()
            # .loc[self.isyns]
            # .values
        )
        return result

    def excludes(
            self,
            label: str = None,
            meta: str = None,
    ) -> Series[bool]:
        return ~self.includes(label, meta)

    def synonymous(
            self,
            label: str | list[str]
    ) -> Series[bool]:
        if isinstance(label, str):
            label = [label]
        isyn = self.synonyms.isyn.loc[label]
        loc = self.isyn.isin(isyn)
        return loc

    @magic.column
    def combo(self):
        result = (
            self.prompts.combo
            .set_axis(self.prompts.irephrase)
            .loc[self.irephrase]
            .values
        )
        return result

    # @magic.column
    # def isyns(self) -> Series[tuple[int]]:
    #     """An ordered tuple of the isyns associated with the combo"""
    #     _ = self.isyn
    #     result = (
    #         self
    #         .reset_index()
    #         .sort_values('isyn')
    #         .groupby('irephrase', sort=False)
    #         .isyn
    #         .apply(tuple)
    #         .loc[self.irephrase]
    #         .values
    #     )
    #     for t in result:
    #         assert all(t[i] <= t[i + 1] for i in range(len(t) - 1))
    #     return result

