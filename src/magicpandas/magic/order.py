from __future__ import annotations

from pandas.core.generic import NDFrame
from functools import cached_property

if False:
    from magicpandas.magic.abc import ABCMagic

if False:
    from .magic import Magic

from typing import TypeVar
T = TypeVar('T')
class Order(str):
    # todo: eventually optimize
    __magic__: ABCMagic = None
    __hash__ = str.__hash__
    first: Order
    second: Order
    third: Order
    synonyms = {
        '__first__': '__first__',
        '__second__': '__second__',
        '__third__': '__third__',
        '__fourth__': '__fourth__',
        'first': '__first__',
        'second': '__second__',
        'third': '__third__',
        'fourth': '__fourth__',
        1: '__first__',
        2: '__second__',
        3: '__third__',
        4: '__fourth__',
    }

    def __call__(self, value):
        value = self.synonyms[value]
        return self.__class__(value)

    # todo: what about when used as class attr?

    def __get__(self: T, outer: Magic, Outer) -> T:
        """
        code that sometimes runs when accessing 3rd order
        """
        if (
                outer is None
                or Outer is Order
        ):
            return self
        key = self.__name__
        cache = outer.__cache__
        if key in cache:
            return cache[key]
        return self

    def __set__(self, outer: Magic, value):
        outer.__cache__[self.__name__] = value

    def __delete__(self, outer: Magic):
        key = self.__name__
        cache = outer.__cache__
        if key in cache:
            del cache[key]

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __int__(self):
        match self:
            case '__first__':
                return 1
            case '__second__':
                return 2
            case '__third__':
                return 3
            case _:
                raise NotImplementedError

    def __index__(self):
        match self:
            case '__first__':
                return 0
            case '__second__':
                return 1
            case '__third__':
                return 2
            case _:
                raise NotImplementedError

    def __next__(self):
        match self:
            case '__first__':
                return Order.second
            case '__second__':
                return Order.third
            case '__third__' | None:
                raise ValueError

    def __eq__(self, other):
        synonyms = self.synonyms
        if other not in synonyms:
            return False
        return str.__eq__(self, synonyms[other])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return int(self) > Order(other).__int__()


Order.first = Order('__first__')
Order.second = Order('__second__')
Order.third = Order('__third__')
Order.fourth = Order('__fourth__')

if __name__ == '__main__':
    assert Order.first == '__first__'
    assert Order.second == '__second__'
    assert Order.third == '__third__'
    assert Order.fourth == '__fourth__'
    assert Order.third != 2
    assert Order.third == 3
    assert Order.third != 1
