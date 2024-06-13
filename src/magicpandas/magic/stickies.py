from __future__ import annotations

from collections import UserDict
from typing import *

from magicpandas.magic.abc import ABCMagic

if False:
    from .magic import Magic


class Stickies:
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
                for key, value in base.__stickies__.items()
            }
            result.update({
                key: value.__sticky__
                for key, value in owner.__dict__.items()
                if isinstance(value, ABCMagic)
            })
            self.__cache__[owner] = result
        else:
            result = self.__cache__[owner]

        return result


if __name__ == '__main__':
    from magicpandas.magic.magic import Magic
    Magic.__stickies__
    Magic.__stickies__
    Magic.__stickies__
    Magic.__stickies__
