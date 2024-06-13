from __future__ import annotations

from typing import *

from magicpandas.magic.abc import ABCMagic

if False:
    from .magic import Magic


class Sticky(FrozenSet):
    def __get__(self, instance: Direction, owner) -> Self:
        name = self.__name__
        cache = instance.__dict__
        if name not in cache:
            Owner = instance.__Owner__
            stickies = Owner.__stickies__
            result = {
                key
                for key in instance
                if stickies.get(key, False)
            }
            result = self.__class__(result)
            cache[name] = result
        return cache[name]

    def __set_name__(self, owner, name):
        self.__name__ = name


class Direction(FrozenSet):
    sticky = Sticky()
    __Owner__: type[Magic]

    def __get__(self, instance: Propagating, owner) -> Self:
        direction = self.__name__
        cache = instance.__dict__
        owner = instance.__Owner__
        if direction not in cache:
            def explore(owner: type[Magic, ABCMagic], direction: str):
                yield from (
                    key
                    for key, value in owner.__directions__.items()
                    if
                    value == direction
                    or value == 'diagonal'
                )

            result = {
                name
                for base in owner.__bases__[::-1]
                if issubclass(base, ABCMagic)
                for name in explore(base, direction)
            }
            result.update(explore(owner, direction))
            result = self.__class__(result)
            cache[direction] = result
            result.__Owner__ = owner
            return result

        return cache[direction]

    def __set_name__(self, owner, name):
        self.__name__ = name


class Propagating(FrozenSet):
    horizontal = Direction()
    vertical = Direction()
    diagonal = Direction()
    sticky = Sticky()
    __cache__: dict[type[Magic], Self] = {}
    __Owner__: type[Magic]

    def __get__(self, instance, owner: type[Magic]) -> Self:
        cache = self.__cache__
        if owner not in cache:
            result = self.__class__(owner.__directions__.keys())
            result.__Owner__ = owner
            cache[owner] = result
        else:
            result = cache[owner]
        return result


if __name__ == '__main__':
    from magicpandas.pandas.frame import Frame

    Frame.__directions__
    Frame.__propagating__.sticky
    Frame.__propagating__
    Frame.__propagating__.vertical
    Frame.__propagating__.horizontal
    Frame.__propagating__.diagonal
    Frame.__propagating__.vertical.sticky
    Frame.__propagating__.horizontal.sticky
    Frame.__propagating__

"""
horizontal & sticky,
horizontal,
vertical & sticky,
vertical
diagonal & sticky,
diagonal

propagating.horizontal.sticky
propagating.vertical
propagating.diagonal.sticky
"""
