from __future__ import annotations
from magicpandas.magic.directions import Directions

import weakref
from typing import *

from magicpandas.magic.order import Order

if False:
    from .magic import Magic

# # @cached.when_set.property
# # @cached.property
# @cached.property
# def __third__(self) -> Self | ABCMagic | NDFrame:
#     """
#     Non-commutative data
#     a.b != a.b (across processes)
#
#     The third order instance of the hierarchy
#
#     class Outer(magic):
#         inner = Inner()
#     class Owner(frame):
#         outer = Outer()
#     Owner = Owner()
#
#     owner.outer.inner
#
#     first.second.third:
#     name            first   second  third
#     cls.order       3       2       3
#     instance.order  3       1       3
#
#     Here, instantiating Owner activates the process and contains live data.
#     `owner` is a third-order instance with live data, and `owner.outer` is
#     a `magic` instance, which is order 1 and still just metadata, but `owner.outer.inner`
#     is a `frame` instance, and contains live data, and is order 3.
#     """
#     if self.__class__.__order__ == 3:
#         return self

class Third:
    def __set_name__(self, owner, name):
        owner.__directions__[name] = 'diagonal'
        self.__name__ = name


    def __get__(self, instance: Magic, owner) -> Optional[Magic] | Self:
        if instance is None:
            return self
        key = self.__name__
        cache = instance.__cache__
        if instance.__order__ == Order.third:
            cache[self.__name__] = weakref.ref(instance)
            return instance
        outer = instance.__outer__
        if outer is None:
            cache[key] = None
            return None

        if key not in cache:
            result = getattr(outer, key)
            cache[key] = weakref.ref(result)
        else:
            result = cache[key]
            if isinstance(result, weakref.ref):
                result = result()
        return result

    def __set__(self, instance: Magic, value):
        if value is not None:
            value = weakref.ref(value)
        instance.__cache__[self.__name__] = value

    def __delete__(self, instance: Magic):
        try:
            del instance.__cache__[self.__name__]
        except KeyError:
            ...
