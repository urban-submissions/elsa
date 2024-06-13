from __future__ import annotations

from functools import *
from numpy import ndarray
from pandas import Series, DataFrame
from typing import *

import magicpandas as magic

if False:
    from elsa.root import Elsa
    from elsa.annotation.unique import Unique


class Check(magic.column):
    """Column is being used as a check for invalid combinations"""


check = Union[Check, Series, ndarray]
globals()['check'] = Check


class Checks:
    def __set_name__(self, owner, name):
        self.__cache__: dict[type[Invalid], set[str]] = {}
        self.__name__ = name

    def __get__(self, instance, owner: type[Invalid]) -> set[str]:
        cache = self.__cache__
        if owner not in cache:
            invalid = {
                key
                for base in owner.__bases__
                for cls in reversed(base.mro())
                if issubclass(cls, Invalid)
                for key in cls.checks
            }
            invalid.update(
                key
                for key, value in owner.__dict__.items()
                if isinstance(value, Check)
            )
            cache[owner] = invalid
        return cache[owner]


class Invalid(magic.Frame):
    __outer__: Unique
    __owner__: Unique
    checks = Checks()

    def __from_inner__(self) -> Self:
        """Called when accessing Unique.invalid to instantiate Invalid"""
        index = self.__outer__.index.unique()
        result = DataFrame(index=index)
        return result

    @magic.index
    def isyns(self):
        ...

    @cached_property
    def unique(self):
        return self.__outer__

    @cached_property
    def elsa(self) -> Elsa:
        return self.__outer__.__outer__.__outer__

    @cached_property
    def label2isyn(self) -> dict[str, int]:
        return self.elsa.synonyms.isyn.to_dict()

    def includes(
            self,
            label: str = None,
            meta: str = None,
    ) -> magic[bool]:
        """
        Determine if a combo includes a label
        Returns ndarrays to save time on redundant alignment
        """
        result = self.unique.includes(label, meta)
        return result

    def excludes(
            self,
            label: str = None,
            meta: str = None,
    ) -> magic[bool]:
        """Determine if a combo excludes a label"""
        result = ~self.includes(label, meta)
        return result

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """ For a given mask, count unique labels"""
        # self.__outer__.get_nunique_labels(loc)
        if isinstance(loc, Series):
            loc = loc.values
        result = (
            self.unique
            .get_nunique_labels(loc)
            .reindex(self.index, fill_value=0)
        )
        return result

    @check
    def c(self) -> check:
        """ multiple conditions """

        # invalid where multiple conditions
        loc = self.unique.meta == 'condition'
        result = self.get_nunique_labels(loc) > 1

        return result

    @check
    def alone_crosswalk(self):
        """alone and crossing crosswalk"""
        a = self.includes('alone')
        a &= self.includes('crossing crosswalk')
        a &= self.includes('sitting')

        # invalid only if condition, state, and activity
        b = self.includes(meta='condition')
        b &= self.includes(meta='state')
        b &= self.includes(meta='activity')

        result = a & b
        return result

    @check
    def sitting_and_standing(self):
        """sitting and standing"""
        a = self.includes('sitting')
        a &= self.includes('standing')
        a &= self.includes('person')
        return a

    @check
    def missing_standing_sitting(self):
        """missing standing or sitting"""
        a = self.excludes('sitting')
        a &= self.excludes('standing')
        b = self.includes('vendor')
        b |= self.includes('shopping')
        b |= self.includes('load/unload packages from car/truck')
        b |= self.includes('waiting in bus station')
        b |= self.includes('working/laptop')

        # invalid only if condition, state, and activity
        c = self.includes(meta='condition')
        c &= self.includes(meta='state')
        c &= self.includes(meta='activity')

        return a & b & c

    @check
    def standing_sitting(self) -> check:
        """alone and multiple states"""

        # invalid where alone and more than 1 state
        a = self.includes('alone')
        loc = self.unique.meta == 'state'
        a &= self.get_nunique_labels(loc) > 1

        return a

    @check
    def couple(self) -> check:
        """ couple and more than 2 states """
        # invalid where couple and more than 2 states
        result = self.includes('couple')
        loc = self.unique.meta == 'state'
        result &= self.get_nunique_labels(loc) > 2
        return result

    @check
    def no_state(self):
        """no state in combo if anything other than condition"""
        a = self.includes(meta='activity')
        a |= self.includes(meta='others')
        a &= self.excludes(meta='state')
        a &= self.excludes('pet')
        return a

    @check
    def no_condition(self):
        """no condition in combo"""
        a = self.excludes(meta='condition')
        a &= self.excludes('pet')
        return a

    @property
    def all_checks(self) -> Self:
        for check in self.checks:
            getattr(self, check)
        return self
