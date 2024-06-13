from __future__ import annotations

from functools import cached_property
from numpy import ndarray
from pandas import Series
from typing import *
from typing import Self

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.evaluation.grid import Grid


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


class Invalid(Resource):
    """
    Aligned with grid, invalid checks are True where any of the members
    of the ibox group are invalid with one another.
    """
    __outer__: Grid
    checks = Checks()

    @cached_property
    def stacks(self):
        return self.__outer__.stacks

    def __from_inner__(self) -> Self:
        index = self.__outer__.ibox.unique()
        result = self(index=index)
        return result

    @check
    def sitting_standing(self) -> magic[bool]:
        """sitting and standing cannot coexist"""
        stacks = self.stacks
        result = stacks.includes('sitting')
        result &= stacks.includes('standing')
        return result

    @check
    def alone(self):
        """alone cannot coexist with > 1 state"""
        stacks = self.stacks
        result = stacks.includes('alone')
        loc = stacks.meta == 'state'
        result &= stacks.get_nunique(loc) > 1
        return result

    @check
    def couple(self):
        """couple cannot coexist with > 2 states"""
        stacks = self.stacks
        result = stacks.includes('couple')
        loc = stacks.meta == 'state'
        result &= stacks.get_nunique(loc) > 2
        return result

    @cached_property
    def grid(self) -> Grid:
        return self.__outer__

    @cached_property
    def rephrase(self):
        return self.grid.rephrase

    @cached_property
    def prompts(self):
        return self.grid.prompts

    @cached_property
    def elsa(self):
        return self.__outer__.elsa
