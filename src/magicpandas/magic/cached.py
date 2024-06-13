from __future__ import annotations

import copy
import inspect
import weakref
from functools import *
from typing import *

from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.from_outer import FromOuter

if False:
    from .magic import Magic


def __get__(self, instance: Magic, owner: type[Magic]):
    self.__Outer__ = owner
    if instance is None:
        return self
    key = self.__from_outer__.__name__
    cache = instance.__cache__
    if key not in cache:
        value = (
            self
            .__from_outer__
            .__get__(instance, owner)
            .__call__()
        )
        self.__set__(instance, value)
    result = cache[key]
    if isinstance(result, weakref.ref):
        result = result()
        if result is None:
            raise ValueError(
                f"weakref to {key} in {instance} is None"
            )
    return result


class Base:
    locals()['__get__'] = __get__
    __from_outer__ = FromOuter()
    __direction__ = 'horizontal'
    __Outer__: type[Magic] = None

    def __init__(self, func):
        self.__from_outer__ = func

    def __set__(self, outer: Magic, value):
        if isinstance(value, ABCMagic):
            value = weakref.ref(value)
            if value() is None:
                raise ValueError(
                    f"weakref to {self.__name__} in {outer} is None"
                )
        # outer.__dict__[self.__name__] = value
        outer.__cache__[self.__name__] = value

    def __delete__(self, outer: Magic):
        try:
            # del outer.__dict__[self.__name__]
            del outer.__cache__[self.__name__]
        except KeyError:
            ...

    def __set_name__(self, owner: Magic, name):
        self.__name__ = name
        self.__Owner__ = owner
        if hasattr(self.__from_outer__, '__set_name__'):
            self.__from_outer__.__set_name__(owner, name)
        owner.__directions__[name] = self.__direction__

    def __repr__(self):
        try:
            return (
                f"{self.__class__.__name__} "
                f"{self.__Owner__.__name__}.{self.__name__}"
            )
        except AttributeError:
            return super().__repr__()

    @property
    def __wrapped__(self):
        return self.__from_outer__

    @__wrapped__.setter
    def __wrapped__(self, value):
        if not inspect.isfunction(value):
            @wraps(self.__from_outer__)
            def wrapper(*args, **kwargs):
                return value
            value = wrapper
        self.__from_outer__ = value

    # todo: no idea what I was doing here? this is a problem for series
    # def __ior__(self, other) -> Self:
    #     result = copy.copy(self)
    #     @wraps(self.__from_outer__)
    #     def wrapper(*args, **kwargs):
    #         return other
    #     result.__from_outer__ = wrapper
    #     return result



def __get__(self, instance: Magic, owner: type):
    if instance is None:
        return self
    if instance.__root__ is not None:
        instance = instance.__root__

    return super(Root, self).__get__(instance, owner)


class Root(Base):
    locals()['__get__'] = __get__

    # stores in root attrs
    def __set__(self, instance, value):
        if instance.__root__ is not None:
            instance = instance.__root__
        super().__set__(instance, value)

    def __delete__(self, instance: Magic):
        if instance.__root__ is not None:
            instance = instance.__root__
        super().__delete__(instance)


def __get__(self: Volatile, instance: Magic, owner):
    if instance is None:
        return self
    key = self.__from_outer__.__name__
    cache = instance.__volatile__
    if key not in cache:
        value = (
            self
            .__from_outer__
            .__get__(instance, owner)
            .__call__()
        )
        self.__set__(instance, value)
    result = cache[key]
    if isinstance(result, weakref.ref):
        result = result()
        if result is None:
            trace = instance.__trace__
            order = instance.__order__
            # second = instance.__second__
            # second = instance.__cache__['__second__']
            raise ValueError(
                f"Weakref to {key} in {trace=} {order=} is None"
            )
    return result


class Volatile(Base):
    """
    properties can change while something like __from_outer__
    or __from_inner__ is run; volatile properties are locally
    cached before a process is run so that they may be renewed
    after the process is done
    """

    locals()['__get__'] = __get__

    def __set__(self, instance: Magic, value):
        cache = instance.__volatile__
        key = self.__name__
        if isinstance(value, ABCMagic):
            value = weakref.ref(value)
            if value() is None:
                raise ValueError(
                    f"weakref to {key} in {instance} is None"
                )
        cache[key] = value

    def __delete__(self, instance: Magic):
        cache = instance.__volatile__
        key = self.__name__
        if key in cache:
            del cache[key]


class Diagonal(Base):
    __direction__ = 'diagonal'


class cached:
    property = Base

    class root:
        property = Root

    class base:
        property = Base

    class volatile:
        property = Volatile

    class diagonal:
        property = Diagonal
