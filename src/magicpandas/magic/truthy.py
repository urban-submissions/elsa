from __future__ import annotations

import copy
import inspect
from types import *
from typing import Self
from typing import TypeVar
T = TypeVar('T')

if False:
    from magicpandas import Magic

__all__ = ['Truthy', 'truthy', ]


class Truthy:
    """Wrapper to allow disabling a function so that bool(func) == False"""
    __name__: str = ''
    __Outer__: type[Magic] = None
    __func__ = None
    __enabled__ = False

    @property
    def __wrapped__(self):
        return self.__func__

    @__wrapped__.setter
    def __wrapped__(self, value):
        cls = self.__Outer__
        cp = copy.copy(self)
        if isinstance(value, bool):
            cp.__enabled__ = value
        else:
            raise TypeError(f'Expected bool, got {type(value)}')
        setattr(cls, self.__name__, cp)

    def __get__(self, outer: Magic, Outer: type[Magic]) -> Self | MethodDescriptorType:
        if outer is None:
            self.__Outer__ = Outer
            return self
        # outer dict is necessary because truthyfunction might be bound to instance
        self = outer.__dict__.get(self.__name__, self)
        self = copy.copy(self)
        self.__func__ = self.__func__.__get__(outer, Outer)
        return self

    def __set__(self, instance: Magic, value):
        if self.__name__ not in instance.__dict__:
            instance.__dict__[self.__name__] = copy.copy(self)
        self = instance.__dict__[self.__name__]
        if isinstance(value, bool):
            self.__enabled__ = value
        elif inspect.isfunction(value):
            self.__func__ = value
            self.__enabled__ = True
        elif isinstance(value, Truthy):
            self.__dict__.update(value.__dict__)
        else:
            raise TypeError(
                f'Expected bool or function, got {type(value)}'
            )

    def __ior__(self, other) -> Self:
        result = copy.copy(self)
        if isinstance(other, bool):
            result.__enabled__ = other
        elif inspect.isfunction(other):
            result.__func__ = other
            result.__enabled__ = True
        elif isinstance(other, Truthy):
            result = copy.copy(other)
        return result

    def __init__(self, func, bool: bool = False):
        self.__func__ = func
        self.__enabled__ = bool

    def __bool__(self):
        return self.__enabled__

    def __repr__(self):
        return (
            f'{self.__class__.__name__} '
            f'{self.__func__.__module__}.{self.__func__.__name__} '
            f'enabled={self.__enabled__}'
        )

    def __set_name__(self, owner: Magic, name):
        self.__name__ = name
        owner.__directions__[name] = 'diagonal'
        if hasattr(self.__func__, '__set_name__'):
            self.__func__.__set_name__(owner, name)

    def __call__(self, *args, **kwargs):
        result = (
            self
            .__func__
            .__call__(*args, **kwargs)
        )
        return result


def truthy(func: T) -> T | Truthy:
    return Truthy(func)
