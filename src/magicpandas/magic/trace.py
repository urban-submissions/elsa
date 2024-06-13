from __future__ import annotations
from typing import TypeVar
T = TypeVar('T')

import copy
from functools import lru_cache

from typing import Optional, Iterable
from magicpandas.magic.default import Default

if False:
    from magicpandas.magic.magic import Magic


class Trace(str):
    __outer__: Optional[Magic] = None
    __name__: str = '__trace__'

    def __set__(self, instance: Magic, value):
        cache = instance.__cache__
        cache[self.__name__] = value = Trace(value)
        value.__outer__ = instance

    def __delete__(self, instance: Magic):
        cache = instance.__cache__
        try:
            del cache[self.__name__]
        except KeyError:
            ...

    def __get__(self, magic: Magic, owner) -> Trace:
        if magic is None:
            return self
        cache = magic.__cache__
        key = self.__name__
        if key in cache:
            result = cache[key]
            return result
        order = magic.__order__
        match order:
            case 1:
                trace = magic.__name__
            case 2:
                outer = magic.__outer__
                if (
                    outer is None
                    or outer.__first__ is None
                ):
                    trace = magic.__name__
                else:
                    trace = f'{outer.__trace__}.{magic.__name__}'

            case 3:
                second = magic.__second__
                if second is None:
                    trace = self
                else:
                    trace = second.__trace__
            case _:
                raise ValueError(f"Invalid order {order}")

        # Avoid caching during `with magic.default`
        result = Trace(trace)
        if not Default.context:
            cache[key] = result
            result.__outer__ = magic
        return result

    @lru_cache
    def __sub__(self: T, other: T | str) -> Trace:
        """
        'a' - '' == 'a'
        'a.b' - 'a' = 'b'
        'a.b.c' - 'a.b' = 'c'
        """
        if not self.startswith(other):
            raise ValueError(f'{other} not in {self}')
        cls = type(self)
        result = cls(
            self
            .removeprefix(other)
            .removeprefix('.')
        )
        return result

    @lru_cache
    def __add__(self, other: str) -> Trace:
        cls = type(self)
        if not self:
            result = other
        elif not other:
            result = self
        else:
            result = cls(f'{self}.{other}')
        result = cls(result)
        return result

    def __call__(self, keys: Iterable[str]) -> object:
        obj = self.__outer__
        for key in keys:
            obj = getattr(obj, key)
        return obj

    # noinspection PyTypeChecker
    def __getitem__(self, item: str) -> Trace | str:
        if isinstance(item, Trace):
            return self(*str.split(item, '.'))
        return super().__getitem__(item)

    def __setitem__(self, key, value):
        # last split gets set
        key = str(key)
        keys = key.split(".")
        obj = self(keys[:-1])
        setattr(obj, keys[-1], value)

    def __delitem__(self, key):
        # last split gets deleted
        key = str(key)
        keys = key.split(".")
        obj = self(keys[:-1])
        delattr(obj, keys[-1])

    def __set_name__(self, owner: Magic, name):
        self.__name__ = name
        owner.__directions__[name] = 'horizontal'
        owner.__stickies__[name] = True



    visited = set()
