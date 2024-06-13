from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from functools import *
from pandas import Series
from typing import *

import magicpandas as magic
from elsa.synonyms.drop import DROP
from elsa.synonyms.order import ORDER
from elsa.synonyms.prompts import PROMPTS
from elsa.synonyms.meta import META
from elsa.synonyms.natural import NATURAL
from elsa.synonyms.incoco import INCOCO

if False:
    from elsa import Elsa


def syn(synonyms: list[str]) -> list[set[str]]:
    return [
        set(syn.split("; "))
        for syn in synonyms
    ]


# todo: config.json like this?
# [
#     {
#         "synonyms": "a, b, c",
#         "prompts": "a, b",
#         "meta": "activity",
#         "natural": {
#             "a": "A",
#             "b": "B",
#             "c": "C"
#         },
#         "order": {
#             "a": "subject"
#         }
#     }
# ]
SYNONYMS = syn([
    "alone; a person; person; individual; pedestrian",
    "at petrol/gas station; at gas station; at petrol station",
    "biking; cycling; riding bike; bicycling; riding bicycle",
    "construction-workers; laborers; builders; construction workers; construction worker; laborer; construction-worker",
    "couple/2people; two people; two pedestrians; two friends; couple; pair",
    "dining; eating; snacking",
    "GSV car interaction; car interaction",
    "group; people; crowd; gathering; many people",
    "on wheelchair; in wheelchair; using wheelchair; riding wheelchair",
    "pet interactions; with dog; with pet",
    # "phone interaction; using phone; on phone",
    "phone interaction; playing with phone; looking at phone; engaged with phone; on phone; using phone",
    "playing",
    "pushing stroller or shopping cart; pushing stroller; pushing baby carriage; pushing pram; pushing shopping cart; pushing cart",
    "running; jogging; sprinting",
    "shopping; buying; purchasing; browsing",
    "sitting; seated; sitting down; sitting on bench or chair; seated on bench or chair; sitting on bench; sitting on chair",
    "sport activities; sports; playing sports; athletics; athletic activities",
    "standing; standing up",
    "street-vendors; vendors; street seller; merchant; vendor; street-vendor; street vendors; street vendor",
    "taking photo; taking picture; taking photograph; taking image; photographing",
    "talking; chatting; conversing; speaking; communicating; arguing; discussing; debating; conversing; dialoguing",
    "talking on phone; talking on cellphone; chatting on phone; chatting on cellphone; conversing on phone; conversing on cellphone; speaking on phone; speaking on cellphone; communicating on phone; communicating on cellphone",
    "waiting in bus station; waiting at bus station; waiting for bus; waiting for bus at bus station",
    "walking; strolling",
    "with bike; with bicycle; with cycle",
    "with coffee or drinks; with coffee; with drink; with beverage",
    "baby/infant; baby; infant; newborn; toddler",
    "crossing crosswalk; crossing street; crossing road; crossing zebra crossing; crossing pedestrian crossing; crossing road",
    "duplicate; duplicated; duplication",
    "elderly; old; senior; aged; aged person; old person; senior person; elderly person",
    "kid; child",
    "with cane or walker; mobility aids; walking aids; walking stick; walking cane; crutches; wheelchair; walking frame; walking aid; mobility aid",
    "model_hint",
    "multi-label; multiple labels; multiple label",
    "no people",
    "not sure/confusing; not sure; confusing; unsure; uncertain; ambiguous",
    "pet; service dog; guide dog",
    "public service/cleaning; public service; community service; cleaning",
    "riding carriage; riding horse carriage; riding horse cart; riding horse wagon; riding horse buggy; riding horse vehicle",
    "teenager; teen; adolescent; youth",
    "working/laptop; working on laptop; working with laptop; working on computer; working with computer; working on desktop; working with desktop",
    "no interaction",
    'police; law enforcement; police officer; cop',
    'load/unload packages from car/truck; loading packages; unloading packages; loading packages from car; unloading packages from car; loading packages from truck; unloading packages from truck; loading packages from vehicle; unloading packages from vehicle; load packages from car; unload packages from car',
    'reading; reading book; reading newspaper; reading magazine; reading paper; reading document; reading text; reading article; reading journal; reading publication; reading publication',
    'with luggage; with suitcase',
    'waiting for food/drinks; waiting for food; waiting for drink; waiting for drinks; waiting for beverage; waiting for meal; waiting for snack; waiting for coffee; waiting for tea; waiting for food or drink; waiting for food or beverage; waiting for drink or beverage; waiting for meal or snack; waiting for coffee or tea',
    'taking cab/taxi; taking taxi; taking cab',
    'picnic; picnicking',
    'riding motorcycle; motorcycle riding; riding motorbike; motorbike riding; motorcycling; motorbiking',
    'hugging; embracing; cuddling; snuggling',
    "pushing wheelchair"
])


# we use synonyms to unify labels across datasets
#   however there are too many synonyms to be used when
#   generating prompts; we have a subset called prompts that
#   are just for prompt generation
# include the article for natural language generation

class Synonyms(magic.Frame):
    __outer__: Elsa
    __owner__: Elsa
    prompts: Prompts
    drop_list: Self

    @cached_property
    def synonyms(self) -> Series[str]:
        """map isyn to syns"""
        result = (
            self
            .reset_index()
            .set_index('isyn')
            .syn
        )
        return result

    def drop_list(self) -> Self:
        """A subset of synonyms which are meant to be dropped"""
        isyn = self.isyn.loc[DROP]
        loc = self.isyn.isin(isyn)
        result = self.loc[loc]
        return result

    def prompts(self) -> Prompts:
        """
        The subset of synoynms meant to be used for prompt generation;
        These aren't the actual prompts; they are the synonyms meant
        to be used in prompts.
        """

    @magic.index
    def syn(self) -> Series[str]:
        """An undercase label that may be associated with a set of synonyms"""

    @magic.column
    def isyn(self) -> Series[int]:
        """The unique index of the synonym set that the label belongs to"""

    @magic.column
    def meta(self) -> Series[str]:
        """The meta information of the synonym set that the label belongs to"""

    @magic.column
    def imeta(self) -> Series[int]:
        """The unique index of the metaclass of the synonym set that the label belongs to"""

    @cached_property
    def label2isyn(self) -> dict[str, int]:
        return self.isyn.to_dict()

    @magic.column
    def label(self) -> Series[int]:
        ...

    @magic.column
    def is_in_coco(self) -> magic[bool]:
        """Specific label is in COCO dataset"""
        result = self.syn.isin(INCOCO)
        return result

    @magic.column
    def is_like_coco(self) -> magic[bool]:
        """Label is synonymous with a label in COCO dataset"""
        loc = self.is_in_coco
        isyn = self.isyn.loc[loc]
        loc = self.isyn.isin(isyn)
        return loc

    @cached_property
    def isyn2syns(self) -> Series[list[str]]:
        result = (
            self.syn
            .groupby(self.isyn)
            .apply(list)
            # .list(set)
        )
        return result

    @cached_property
    def isyn2ilabel(self) -> Series[str]:
        """
        Map each isyn to an ilabel that is present in the truth;
        if the synonym is not present it is not included here.
        """

    @magic.column
    def iorder(self) -> Series[int]:
        """The natural order of the synonym in the prompt"""
        index = Series(ORDER.keys()).str.casefold()
        order = (
            Series(ORDER.values(), index=index, name='isyn')
            .rename_axis('syn')
        )
        loc = index.isin(self.syn).values
        order = order.loc[loc]

        index = self.isyn.loc[order.index]
        isyn2order = (
            order
            .set_axis(index, axis=0)
            .groupby(level='isyn')
            .first()
        )
        loc = ~self.syn.isin(order.index)
        isyn = self.isyn.loc[loc]
        syn = self.syn[loc]
        appendix = isyn2order.loc[isyn].set_axis(syn, axis=0)
        # categorical dtype where subject < prepositional < verb
        categories = 'subject prepositional verb'.split()
        dtype = pd.CategoricalDtype(categories, True)
        result = (
            pd.concat([order, appendix])
            .astype(dtype)
            .loc[self.syn]
            # .values
        )
        return result

    def __from_inner__(self) -> Self:
        repeat = np.fromiter(map(len, SYNONYMS), int, len(SYNONYMS))
        isyn = np.arange(len(SYNONYMS)).repeat(repeat)
        count = repeat.sum()
        it = itertools.chain.from_iterable(SYNONYMS)
        syn = np.fromiter(it, dtype=object, count=count)
        index = (
            Series(syn, name="syn")
            .str.casefold()
            .pipe(pd.Index)
        )
        assert not index.duplicated().any()
        result: Self = self({
            "isyn": isyn,
        }, index=index)

        meta = Series(META, name='meta')
        assert not meta.index.duplicated().any()
        # assert not result.index.duplicated().any()

        loc = meta.index.isin(result.index)
        meta = meta.loc[loc]
        isyn = result.isyn.loc[meta.index]
        meta = meta.set_axis(isyn)

        loc = result.isyn.isin(meta.index)
        if not loc.all():
            missing = result.syn.values[~loc].tolist()
            formatted_missing = "\n".join(missing)
            raise ValueError(
                f'The following synonyms are missing from '
                f'the synonym-to-meta mapping:\n{formatted_missing}'
            )
        # meta = meta.index
        loc = ~meta.index.duplicated()
        meta = meta.loc[loc]
        result['meta'] = meta.loc[result.isyn].values

        # fallback on first isyn if syn not in meta

        return result

    @magic.column
    def natural(self) -> magic[str]:
        result = Series(self.syn, index=self.syn, name='natural')
        natural = Series(NATURAL, name='natural')
        loc = ~self.syn.isin(natural.index)
        if loc.any():
            eg = self.syn[loc].tolist()
            msg = f'The following labels are not in the natural metadata: {eg}'
            self.__logger__.info(msg)
        result.update(natural)
        return result

    @magic.column
    def cardinal(self) -> magic[str] | str:
        """Choose an arbitrary label to represent synonyms"""
        result = (
            self.natural
            .groupby(self.isyn.values)
            .first()
            .loc[self.isyn]
            .values
        )
        return result


class Prompts(Synonyms):
    __outer__: Synonyms

    @magic.column
    def natural(self) -> Series[str]:
        """The natural language description of the synonym set"""
        result = (
            Series(NATURAL)
            .loc[self.syn]
            .values
        )
        return result

    def __from_inner__(self) -> Self:
        synonyms = self.__outer__
        repeat = np.fromiter(map(len, PROMPTS), int, len(PROMPTS))
        count = repeat.sum()
        it = itertools.chain.from_iterable(PROMPTS)
        syn = np.fromiter(it, dtype=object, count=count)
        syn = pd.Index(syn, name='syn')
        loc = synonyms.syn.isin(syn)
        result = synonyms.loc[loc]
        return result
