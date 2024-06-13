from __future__ import annotations
import json
import collections


import collections
import json
import pprint

if False:
    from magicpandas.magic.magic import Magic

class Config(collections.UserDict):

    def __repr__(self):
        return pprint.pformat(self.data, indent=4, width=1)

    copies = False

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __set__(self, instance: Magic, value):
        instance.__cache__[self.__name__] = value

    def __delete__(self, instance: Magic):
        try:
            del instance.__cache__[self.__name__]
        except KeyError:
            ...

    def __get__(self, instance: Magic, owner):
        if instance is not None:
            instance = instance.__third__
        if instance is None:
            return self
        key = self.__name__
        cache = instance.__cache__
        if key in cache:
            return cache[key]
        owner = instance.__owner__

        if owner is None:
            value = self.__class__()
        else:
            value = owner.__config__
        cache[key] = value

        return value




# class Config(collections.UserDict):
#     copies = False
#
#     def __set_name__(self, owner, name):
#         self.__name__ = name
#
#     def __get__(self, instance: Magic, owner):
#         if instance is not None:
#             ...
#
#
#
#     def __set__(self, instance: Magic, value):
#         instance.__cache__[self.__name__] = value
#
#     def __delete__(self, instance: Magic):
#         del instance.__cache__[self.__name__]
#
#
#     def __repr__(self):
#         return super().__repr__()
#
