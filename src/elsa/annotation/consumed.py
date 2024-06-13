from __future__ import annotations

import numpy as np

import magicpandas as magic

if False:
    pass

from typing import *
from elsa.annotation.rephrase import Rephrase
from dataclasses import dataclass

if False:
    from elsa.annotation.rephrase import Rephrase

consuming = []


@dataclass
class consume:
    consumer: str | list[str]
    consumed: str | list[str]
    func = None

    def __post_init__(self):
        consuming.append(self)
        self.consumer = self.consumer.split(', ')
        self.consumed = self.consumed.split(', ')

    def __call__(self, func):
        self.func = func
        return func

    @property
    def staticmethod(self):
        return self


class Walking:
    @consume(
        consumer='walking',
        consumed='standing',
    )
    def walking(self: str, consumed: list[str]):
        return self

class Cart:
    # @consume(
    #     consumer="pushing cart",
    #     consumed="walking",
    # )
    # def pushing_cart(self: str, consumed: list[str]):
    #     return self

    @consume(
        consumer='standing, sitting',
        consumed='pushing cart',
    )
    def with_cart(self: str, consumed: list[str]):
        return f'{self} with a shopping cart'


class Stroller:

    # @consume(
    #     consumer="pushing stroller",
    #     consumed="walking",
    # )
    # def pushing_stroller(self: str, consumed: list[str]):
    #     return self

    @consume(
        consumer='standing, sitting',
        consumed='pushing stroller',
    )
    def with_stroller(self: str, consumed: list[str]):
        return f'{self} with stroller'

    @consume(
        consumer='pushing stroller',
        consumed='child, baby',
    )
    def stroller(self: str, consumed: list[str]):
        return self


class Crosswalk:
    @consume(
        consumer='crossing crosswalk',
        consumed='standing, walking, running',
    )
    def crossing_crosswalk(self: str, consumed: list[str]):
        result = f"{'and '.join(consumed)} to cross a crosswalk"
        return result


class OtherPerson:

    @consume(
        consumer='kid, teenager, elderly, police, laborer, vendor',
        consumed='person',
    )
    def person(self: str, consumed: list[str]):
        return self
        # return f'{self} alone'


    @consume(
        consumer='couple',
        consumed='kid, teenager, elderly, police, laborer, vendor',
    )
    def couple(self: str, consumed: list[str]):
        return f'{self} including {"and ".join(consumed)}'

    @consume(
        consumer='group',
        consumed='kid, teenager, elderly, police, laborer, vendor',
    )
    def group(self: str, consumed: list[str], ):
        return f"{self} including {'and '.join(consumed)}"
        # todo: probably make a new magic frame, that encapsulates these tuples
        # yield len(self), 'condition'
        # yield 'including', None
        # yield 'and '.join(consumed), 'other'



class MobilityAid:
    @consume(
        consumer='walking, standing',
        consumed='mobility aid',
    )
    def walking(self: str, consumed: list[str]):
        return f'{self} with a cane'

class Wheelchair:
    @consume(
        consumer='sitting',
        consumed='on wheelchair',
    )
    def sitting(self: str, consumed: list[str]):
        return f'{self} on a wheelchair'



class PublicServices:
    ...


class Sports:

    @consume(
        consumer='sports',
        consumed='standing, walking, running',
    )
    def sports(self: str, consumed: list[str]):
        """Sports implies standing, walking, running"""
        return self


class Consumed(Rephrase):
    __outer__: Rephrase

    def __from_inner__(self) -> Self:
        """Called when accessing Rephrase.consumed to instantiate Consumed"""
        rephrase = self.__outer__
        _ = rephrase.natural, rephrase.isyns, rephrase.isyn
        names = rephrase.index.names
        result = rephrase.reset_index('isyns')
        result['satiated'] = False
        inatural = result.columns.get_loc('natural')
        isatiated = result.columns.get_loc('satiated')
        label2natural = self.synonyms.natural.to_dict()

        for consume in consuming:
            for consumer in consume.consumer:
                # select where includes consumer
                a = (
                    result
                    .includes(consumer)
                    .loc[result.isyns]
                    .values
                )

                # select where includes consumed
                b = np.zeros_like(a, bool)
                for consumed in consume.consumed:
                    b |= (
                        result
                        .includes(consumed)
                        .loc[result.isyns]
                        .values
                    )

                c = a & b
                if not c.any():
                    continue

                d = result.synonymous(consumer).values & c
                e = result.synonymous(consume.consumed).values & c
                d &= ~e
                e &= ~d

                ilocs = (
                    result
                    .assign(iloc=np.arange(len(result)))
                    .loc[d]
                    .groupby(level='irephrase', sort=False)
                    .iloc
                    .first()
                )
                # get a list of all unique consumed
                list_consumed = (
                    result
                    .loc[e]
                    .groupby('irephrase', sort=False)
                    .natural
                    .unique()
                )
                assert len(list_consumed) == len(ilocs)

                for iloc, consumed in zip(ilocs, list_consumed):
                    if result.satiated.iloc[iloc]:
                        continue
                    label = result.label.iloc[iloc]
                    natural = label2natural[label]
                    natural = consume.func(natural, consumed)
                    result.iloc[iloc, inatural] = natural
                    result.iloc[iloc, isatiated] = True

                result = result.loc[~e].copy()

        result: Self = (
            result
            .reset_index()
            .set_index(names)
        )
        assert rephrase.isyns.isin(result.isyns).all()
        return result

    @magic.column
    def satiated(self) -> bool:
        return False
