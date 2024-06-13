from __future__ import annotations

import inspect
from typing import TypeVar
T = TypeVar('T')
O = TypeVar('O')

if False:
    from magicpandas.magic.magic import Magic


class __setter__:
    def __get__(self, instance: object, owner):
        if instance is None:
            return self
        key = self.__name__
        if key in instance.__dict__:
            return (
                instance.__dict__
                [key]
                .__get__(instance, owner)
            )
        return self

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(self, instance: object, value):
        instance.__dict__[self.__name__] = value

    def __call__(self, value):
        return value


class Cached:
    """
    The PyCharm IDE is hard-coded to recognize anything with wrapped
    with @property as returning the return type of the function
    when the method is accessed from an instance. Here we get around
    this by assigning property as an attribute, allowing
    @cached.property to work the same as @property, even though
    it's not a property.
    """
    __Self__: type[Magic] = None
    __setter__ = __setter__()

    def __get__(self, instance: Magic, owner: type[Magic]):
        self.__Self__ = owner
        if instance is None:
            return self
        key = self.__from_outer__.__name__
        cache = instance.__dict__
        if key not in cache:
            value = (
                self
                .__from_outer__
                .__get__(instance, owner)
                .__call__()
            )
            self.__set__(instance, value)
        result = cache[key]
        return result

    def __repr__(self):
        return (
            f"{self.__class__.__name__} "
            f"{self.__Self__.__name__}.{self.__name__}"
        )

    # # noinspection PyArgumentList
    def __init__(self, *args, **kwargs):
        if args and inspect.isfunction(args[0]):
            func, *args = args
            self.__from_outer__ = func

    def __set__(self, outer: Magic, value):
        # if isinstance(value, ABCMagic):
        #     value = weakref.ref(value)
        outer.__dict__[self.__from_outer__.__name__] = value

    def __delete__(self, outer: Magic):
        try:
            del outer.__dict__[self.__from_outer__.__name__]
        except KeyError:
            ...

    def __set_name__(self, owner: type[Magic], name):
        self.__name__ = name
        if hasattr(self.__from_outer__, '__set_name__'):
            self.__from_outer__.__set_name__(owner, name)
        owner.__propagating__.add(name)

    def __map_option__(self: T, option: O, ) -> T | O:
        return option


class Root(Cached):
    # stores in root attrs
    def __get__(self, instance: Magic, owner: type):
        if instance is None:
            return self
        if instance.__root__ is not None:
            instance = instance.__root__

        return super().__get__(instance, owner)

    def __set__(self, instance, value):
        if instance.__root__ is not None:
            instance = instance.__root__
        super().__set__(instance, value)

    def __delete__(self, instance: Magic):
        if instance.__root__ is not None:
            instance = instance.__root__
        super().__delete__(instance)


# class CmdLine(Cached):
#     # todo: things marked with cline should be able to be set in the commandline
#     ...




# class Nearest(Cached):
# # @functools.cached_property
# # def cls(self):
# #     return (
# #         inspect
# #         .getfullargspec(self.__from_outer__)
# #         .annotations['return']
# #     )
#
# def __init__(self, *args, **kwargs):
#     super().__init__(*args, **kwargs)
#     cls = self.cls
#     if isinstance(cls, str):
#         raise TypeError(
#             f'NearestCached requires a type annotation for {self.__from_outer__}; '
#             f'got forward reference {cls} instead of a type. '
#         )
#
#
# def __get__(self, instance: Magic, owner):
#     self.__Self__ = owner
#     if instance is None:
#         return self
#     key = self.__from_outer__.__name__
#     cache = instance.__dict__
#     if key not in cache:
#         cls = self.cls
#         value = instance.__outer__
#         while not isinstance(value, cls):
#             if value is None:
#                 raise AttributeError
#             value = value.__outer__
#         self.__set__(instance, value)
#     result = cache[key]
#     if isinstance(result, weakref.ref):
#         result = result()
#     return result


# class StaticCached(Cached):
#     # static does not set metadata on itself, and only uses attributes
#     # first, second, third
#
#     """
#     PropagatingCached is a simpler version of Cached to avoid three problems:
#     1. RecursionError from trying to use metadata in metadata propagation,
#     2. RecursionError from pandas __finalize__ deepcopying attributes
#         e.g. owner.attrs[inner] == self and inner.attrs[owner] == self
#     3. ReferenceError from propagated attrs holding long-gone references
#
#     PropagatingCached is the only appropriate `cached.property` for weakrefs.
#     """


# class WeakrefCached(PropagatingCached):
#
#     def __get__(self, outer: object, Outer):
#         key = self.__key__
#         if key not in outer.__dict__:
#             outer.__dict__[key] = Proxy(
#                 self
#                 .__from_outer__
#                 .__get__(outer, Outer)
#                 .__call__()
#             )
#         result = outer.__dict__[key]
#         return result
#
#     def __set__(self, instance, value):
#         instance.__dict__[self.__name__] = Proxy(value)
#
#     def __delete__(self, instance):
#         try:
#             del instance.__dict__[self.__name__]
#         except KeyError:
#             ...
#

# class WhenSet(Cached):
#     def __get__(self, instance: Magic, owner: type):
#         if instance is None:
#             return self
#         key = self.__from_outer__.__name__
#         cache = instance.__dict__
#         if key not in cache:
#             result = (
#                 self
#                 .__from_outer__
#                 .__get__(instance, owner)
#                 .__call__()
#             )
#         else:
#             result = cache[key]
#         return result
#


class RootDecorator:
    property = Root


# class CmdLineDecorator:
#     property =


# class WhenSetDecorator:
#     property = WhenSet


class cached:
    property = Cached
    root = RootDecorator
    # cmdline = CmdLineDecorator
    postinit = Cached


    # def property(func: T) -> T | Cached:
    #     ...

    # property = Cached
    # when_set = WhenSetDecorator

# todo
# @cached.outer.property
# def coords(self) -> Coords:
#     ...
# finds the outerleast instance of Coords

class Test:
    @cached.property
    def func(self) -> int:
        ...
