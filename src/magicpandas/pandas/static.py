# from __future__ import annotations
# from __future__ import annotations
#
# from magicpandas.pandas import attr
# from ..magic.static import Static
#
# if False:
#     from .ndframe import NDFrame
#
#
# # todo: but __name__
#
# def __get__[A: Cached](self: A, outer: NDFrame, Outer) -> A:
#     key = self.__key__
#     if key in outer.attrs:
#         return outer.attrs[key]
#     result = self.__from_outer__(outer)
#
#     return result
#
#
# class Cached(
#     attr.Cached,
#     Static
# ):
#     locals()['__get__'] = __get__
#
#     def __set__(self, outer: NDFrame, value):
#         outer.attrs[self.__key__] = value
#
#     def __delete__(self, outer: NDFrame):
#         try:
#             del outer.attrs[self.__key__]
#         except KeyError:
#             ...
#
# attr = Cached
