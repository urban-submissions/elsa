from __future__ import annotations

from typing import *

if False:
    from .magic import Magic


class ClassBoundStrings(set):

    def __set_name__(self, owner: Magic, name):
        self.__cache__: dict[type[Magic], ClassBoundStrings] = {}
        self.__name__ = name

    def __get__(self, instance: object, owner: type[Magic]) -> Self:
        from magicpandas.magic.abc import ABCMagic
        if owner not in self.__cache__:
            result = self.__cache__[owner] = self.__class__(
                name
                for base in owner.__bases__
                if issubclass(base, ABCMagic)
                and hasattr(base, self.__name__)
                for name in getattr(base, self.__name__)
            )
        else:
            result = self.__cache__[owner]
        return result





if __name__ == '__main__':
    test = ClassBoundStrings(v for v in range(4))

