from __future__ import annotations
from magicpandas.magic.directions import Directions

from typing import *

if False:
    from .magic import Magic

class Volatile:
    """
    Volatile attributes are liable to change during the construction
    of a magic object due to other attributes potentially sharing
    the second order magic. It is necessary to copy the volatiles
    at the beginning and reassign them at the end of the construction.
    """
    __direction__ = Directions.diagonal
    def __set_name__(self, owner: type[Magic], name):
        # owner.__directions__[name] = 'diagonal'
        self.__name__ = name

    def __get__(self, instance: Magic, owner: type[Magic]) -> dict[str, Any]:
        return (
            instance.__cache__
            .setdefault(self.__name__, {})
        )

    def __set__(self, instance: Magic, value):
        instance.__cache__[self.__name__] = value

    def __delete__(self, instance: Magic):
        try:
            del instance.__cache__[self.__name__]
        except KeyError:
            ...

