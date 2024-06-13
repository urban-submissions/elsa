from __future__ import annotations

from collections import UserDict
from typing import *

from magicpandas.magic.abc import ABCMagic

if False:
    from .magic import Magic


class Direction(FrozenSet):
    def __set_name__(self, owner, name):
        self.__name__ = name

    def __str__(self):
        return self.__name__

    def __get__(self, instance: Directions, owner) -> Self | str:
        if instance is None:
            return self.__name__
        name = self.__name__
        cache = instance.__dict__
        result = cache[name] = self.__class__(
            key
            for key, value in instance.items()
            if value == name
        )
        result.__name__ = name
        return result


class Directions(UserDict[str, str]):
    horizontal = Direction()
    vertical = Direction()
    diagonal = Direction()
    static = Direction()

    def __set_name__(self, owner: Magic, name):
        self.__cache__: dict[type[Magic], Self] = {}
        self.__name__ = name

    def __get__(self, instance: Magic, owner: type[Magic]):
        if owner not in self.__cache__:
            result = self.__cache__[owner] = self.__class__({
                name: str(direction)
                for base in owner.__bases__[::-1]
                if issubclass(base, ABCMagic)
                   and hasattr(base, self.__name__)
                for name, direction in getattr(base, self.__name__).items()
            })
        else:
            result = self.__cache__[owner]
        return result

    def __setitem__(self, key: str, value: str):
        assert value in ('horizontal', 'vertical', 'diagonal', 'static')
        if value in self.__dict__:
            del self.__dict__[value]
        super().__setitem__(key, value)

    def __delitem__(self, key):
        raise NotImplementedError


