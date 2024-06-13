from __future__ import annotations
from magicpandas.magic.directions import Directions

import inspect
from functools import *

if False:
    from .magic import Magic
    from magicpandas.pandas.ndframe import NDFrame


class FromParams:
    def __set_name__(self, owner: Magic, name):
        owner.__directions__[name] = 'diagonal'
        self.__name__ = name

    def __get__(self, instance: Magic, owner):
        # gets the function
        if instance is None:
            return self
        key = self.__name__
        cache = instance.__dict__
        if key not in cache:
            return None
        # result = (
        #     cache
        #     .__getitem__(key)
        #     .__get__(instance, owner)
        # )
        # return result
        result = (
            cache
            .__getitem__(key)
            # .__get__(instance.__outer__, owner)
            .__get__(instance, owner)
        )
        return result


    def __set__(self, instance: Magic, func):
        parameters = inspect.signature(func).parameters
        argnames = [
            param.name
            for param in parameters.values()
            if param.default == param.empty
        ][1:]
        # skip first
        items = iter(parameters.items())
        next(items)
        defaults = {
            key: param.default
            if param.default != param.empty
            else None
            for key, param in items
        }

        @wraps(func)
        def wrapper(inner: NDFrame, *args, **kwargs):
            # inner.__from_params__ = False
            owner = inner.__owner__
            outer = inner.__outer__
            passed = defaults.copy()
            passed.update(zip(argnames, args))
            passed.update(kwargs)
            cache = owner.__dict__.setdefault(inner.__key__, {})

            key = tuple(passed.values())
            try:
                return cache[key]
            except KeyError:
                hashable = True
            except (IndexError, TypeError):
                hashable = False

            """
            allow for self.__inner__(frame) 
            while also allowing for super().<inner>(args, kwargs)
            so we replace outer.inner with the constructor function
            while super().<inner> will still cause a recursion
            """

            outer.__inner__, inner_ = inner.__constructor__, outer.__inner__
            inner.__owner__, owner_ = outer.__third__, inner.__owner__
            # call func with outer.inner(...)
            inner.__outer__ = outer
            volatile = outer.__volatile__.copy()
            result = func(outer, *args, **kwargs)
            # apply inner metadata with outer.__inner__(...)
            outer.__volatile__.update(volatile)
            result = outer.__inner__(result)
            outer.__inner__ = inner_
            inner.__owner__ = owner_


            # cache result
            if hashable:
                cache[key] = result

            return result

        instance.__dict__[self.__name__] = wrapper

    def __delete__(self, instance: Magic):
        key = instance.__inner__.__key__
        cache = instance.__owner__.__dict__
        del cache[key]



    def __bool__(self):
        return False
