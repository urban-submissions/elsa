from __future__ import annotations
from magicpandas.magic.directions import Directions
from  magicpandas.magic import globals

import weakref
from typing import *

if False:
    from .magic import Magic


class Root:
    """
    Bespoke because we don't want it to cache when instance is root,
    but we want cached when owner != None
    """

    def __set_name__(self, owner: Magic, name):
        self.__name__ = name
        owner.__directions__[name] = 'diagonal'

    def __get__(self, instance: Magic, owner) -> Optional[Magic] | Self:
        if instance is None:
            return self
        # cache = instance.__volatile__
        cache = instance.__cache__
        key = self.__name__
        if key not in cache:
            if instance.__order__ == 3:
                return instance
            else:
                return None
            # return instance
        result = cache[key]
        if isinstance(result, weakref.ref):
            result = result()
        return result

    def __set__(self, instance: Magic, value):
        if value is not None:
            globals.root = value
            value = weakref.ref(value)
        instance.__cache__[self.__name__] = value

    def __delete__(self, instance: Magic):
        try:
            del instance.__cache__[self.__name__]
        except KeyError:
            ...
