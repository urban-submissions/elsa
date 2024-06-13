from __future__ import annotations

from collections import UserDict
from typing import *

from magicpandas.magic.abc import ABCMagic

if False:
    from .magic import Magic


class Delayed:
    def __set_name__(self, owner, name):
        self.__name__ = name
        self.__cache__ = {}

    def __get__(self, instance: Magic, owner: type[Magic]) -> Self:
        base: type[Magic]
        name: str = self.__name__
        if owner not in self.__cache__:
            result = {
                key: value
                for base in owner.__bases__[::-1]
                if issubclass(base, ABCMagic)
                and hasattr(base, name)
                for key, value in base.__delayed__.items()
            }
            result.update({
                key: owner
                for key, value in owner.__annotations__.items()
            })
            self.__cache__[owner] = result
        else:
            result = self.__cache__[owner]

        return result