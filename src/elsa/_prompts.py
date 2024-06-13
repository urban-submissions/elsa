from __future__ import annotations

from collections import defaultdict

import itertools
import numpy as np
import pandas as pd
# from autodistill.detection import CaptionOntology
from functools import *
from pandas import Series
from typing import *
from typing import Self

import magicpandas as magic
from elsa.resource import Resource

if False:
    from .truth import Truth

CaptionOntology = {}
# class Ontologies(magic.Magic):
#     __outer__: Prompts
#     loc: MutableMapping[Any, Self]
#     iloc: MutableMapping[Any, Self]
#
#     @magic.cached.property
#     def per_prompt(self) -> list[CaptionOntology]:
#         """One ontology per unique prompt."""
#         prompts = self.__outer__
#         ont = dict(zip(prompts.caption.values, prompts.cls.values))
#         result = [
#             CaptionOntology({caption: cls})
#             for caption, cls in ont.items()
#         ]
#         return result
#
#     @magic.cached.property
#     def per_isyn(self) -> list[CaptionOntology]:
#         """One ontology per unique isyn."""
#         isyn2dict = defaultdict(dict)
#         prompts = self.__outer__
#         for isyn, caption, cls in zip(
#                 prompts.isyn.values,
#                 prompts.caption.values,
#                 prompts.cls.values,
#         ):
#             isyn2dict[isyn][caption] = cls
#         result = [
#             CaptionOntology(ont)
#             for ont in isyn2dict.values()
#         ]
#         return result
#
#     @magic.cached.property
#     def one(self) -> CaptionOntology:
#         """One ontology for all the prompts."""
#         prompts = self.__outer__
#         ont = dict(zip(prompts.caption.values, prompts.cls.values))
#         result = CaptionOntology(ont)
#         return result
#


"""
elsa.truth.synonyms
"""



class Prompts(Resource):
    # caption and cls docstrings are from autodistillowlv2 docs
    __outer__: Truth
    # ontologies = Ontologies()

    @cached_property
    def ilabel(self):
        """1:many for iprompt:ilabel"""
        ilabels = self.ilabels.values
        count = len(ilabels)
        repeat = np.fromiter(map(len, ilabels), int, count)
        count = repeat.sum()
        it = itertools.chain.from_iterable(ilabels)
        ilabel = np.fromiter(it, int, count)
        iprompt = self.iprompt.repeat(repeat)
        result = Series(ilabel, index=iprompt)
        return result



    @cached_property
    def isyn(self) -> Series[int]:
        """1:many for iprompt:isyn"""
        isyns = self.isyns.values
        count = len(isyns)
        repeat = np.fromiter(map(len, isyns), int, count)
        count = repeat.sum()
        it = itertools.chain.from_iterable(isyns)
        isyn = np.fromiter(it, int, count)
        iprompt = self.iprompt.repeat(repeat)
        result = Series(isyn, index=iprompt)
        return result

    @cached_property
    def ilabel(self) -> Series[int]:
        _ = self.labels.isyn
        result = (
            self.labels
            .groupby('isyn')
            .ilabel
            .first()
            .loc[self.isyn]
            .set_axis(self.isyn.index)
        )
        return result

    @magic.index
    def iprompt(self):
        """Unique index for each prompt"""
        return np.arange(len(self))

    @magic.column
    def isyns(self) -> Series[tuple]:
        """Unique tuple of ordered isyns for each prompt"""

    @magic.column
    def label(self) -> Series[str]:
        """The multilabel generated for each prompt"""

    @magic.column
    def caption(self) -> Series[str]:
        """Caption is the prompt sent to the base model"""
        return self.natural.values

    @magic.column
    def cls(self):
        """
        Class is the label that will be saved for that caption
        in the generated annoations
        """
        result = (
            self.label
            .groupby(self.isyns)
            .first()
            .loc[self.isyns]
            .values
        )
        return result

    @magic.column
    def ilabels(self) -> Series[tuple]:
        """
        Unique tuple of ordered ilabels for each prompt;
        We map the isyn back to the ilabel from the labels resource.
        If in the labels metadata multiple labels have the same isyn,
        we choose the first label.
        """

    @magic.column
    def ontology(self) -> Series[CaptionOntology]:
        result = np.fromiter((
            CaptionOntology({prompt: prompt})
            for prompt in self.natural.values
        ), dtype=object, count=len(self))
        # noinspection PyTypeChecker
        return result

    @magic.column
    def natural(self) -> Series[str]:
        """
        Each label is processed into a more natural form
        that a human would actually use
        """

    def __from_inner__(self) -> Self:
        truth = self.__outer__
        _ = truth.isyn, truth.ibox

        def apply(isyn: Series):
            return tuple(set(isyn))

        # get all unique ordered isyn
        uniques: np.ndarray = (
            truth
            .reset_index()
            .sort_values('ilabel')
            .groupby('ibox', sort=False)
            .isyn
            .apply(apply)
            .unique()
        )
        repeat = np.fromiter(map(len, uniques), int, len(uniques))
        isyns = np.arange(len(uniques)).repeat(repeat)
        isyn = np.fromiter(itertools.chain.from_iterable(uniques), int, repeat.sum())

        labels = (
            self.synonyms.prompts
            .reset_index()
            .groupby('isyn', sort=False)
            .syn
            .apply(list)
            .loc[isyn]
        )
        syn = (
            self.synonyms.prompts
            .reset_index()
            .groupby('isyn', sort=False)
            .syn
            .apply(list)
            .loc[isyn]
        )

        natural = (
            self.synonyms.prompts.natural
            .groupby(self.synonyms.prompts.isyn, sort=False)
            .apply(list)
            .loc[isyn]
            + '.'
        )

        # Series.groupby().prod
        def apply(labels: Series):
            return pd.MultiIndex.from_product(labels.tolist())

        multilabels = (
            labels
            .groupby(isyns, sort=False)
            .apply(apply)
        )
        syns = (
            syn
            .groupby(isyns, sort=False)
            .apply(apply)
        )
        multinatural = (
            natural
            .groupby(isyns, sort=False)
            .apply(apply)
        )


        prompts = [
            ' '.join(pair)
            for multilabel in multilabels
            for pair in multilabel
        ]
        naturals = [
            ' '.join(pair)
            for multinatural in multinatural
            for pair in multinatural
        ]

        repeat = (
            self.synonyms.prompts
            .reset_index()
            .groupby('isyn', sort=False)
            .size()
            .loc[isyn]
            .groupby(isyns, sort=False)
            .prod()
        )
        isyns = uniques.repeat(repeat)

        result = self({
            'isyns': isyns,
            'label': prompts,
            'natural': naturals,
        })
        _ = result.iprompt
        result = result.set_index('iprompt')

        # # iprompt: ilabel is 1:many
        # # iprompt: isyn is 1:many
        # # here we assign magic series instead of magic columns
        # # to support this relation
        # isyns = result.isyns.values
        # repeat = np.fromiter(map(len, isyns), int, len(isyns))
        # index = result.index.repeat(repeat)
        # it = chain.from_iterable(isyns)
        # count = repeat.sum()
        # isyn = np.fromiter(it, int, count)
        # labels = self.labels
        # _ = labels.isyn
        # # map each isyn to first valid ilabel from labels metadata
        # # then get that ilabel for each isyn in the prompt
        # ilabel = (
        #     labels
        #     .drop_duplicates('ilabel')
        #     .set_index('isyn')
        #     .ilabel
        #     .loc[isyn]
        #     .values
        # )

        # result.isyn = Series(isyn, index=index, name='isyn')
        # result.ilabel = Series(ilabel, index=index, name='ilabel')

        return result
