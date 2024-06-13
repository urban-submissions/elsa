from __future__ import annotations
from magicpandas.magic.directions import Directions
from magicpandas.magic.default import Default

import weakref

from magicpandas.magic.order import Order

if False:
    from .magic import Magic


class Second:
    # Second doesn't need to be propagated because it is
    #   inferred from either outer[__name__] or first[__second__]
    def __set_name__(self, owner: Magic, name):
        owner.__directions__[name] = 'diagonal'
        self.__name__ = name

    # problem is that outer changes for first order
    # if order==3, should be kept in self.dict regardless
    def __get__(self, magic: Magic, owner):
        if magic is None:
            return self

        order = magic.__order__
        cache = magic.__cache__
        name = self.__name__
        if (
            order == 3
            and name in cache
        ):
            # raise NotImplementedError
            result = cache[name]()
            if result is None:
                trace = magic.__trace__
                key = magic.__trace__
                raise ValueError(
                    f"Weakref to {key} in {trace=} {order=} is None"
                )
            return result

        outer = magic.__outer__

        # todo: sometimes outer.__outer__ is garbage collected, so we can't access it
        #   to determine if we should store in outer or self. In that case, we assume
        #   it is stored in outer. Is there a cleaner way to do this?

        try:
            if outer is not None:
                _ = outer.__outer__
        except ValueError:
            # return outer.__second__.__dict__[magic.__name__]
            try:
                return outer.__second__.__dict__[magic.__name__]
            except KeyError as e:
                # just raise it, no extra msg
                # todo: how do we get here?
                #   very hard bug to understand, might be because outer
                #   is changing; for now just preemptively access columns
                raise e
                # todo: resolve this; why is it necessary to include this?
                # dict = outer.__second__.__dict__
                # key = magic.__name__
                # result = magic.__class__()
                # result.__order__ = Order.second
                # magic.__first__.__propagate__(result)
                # dict[key] = result
                #
                #

        # if (
        #         # Magic.magic
        #         outer is None
        # ) or (
        #         # magic.magic
        #         # outer.__outer__ is None
        #         # and outer.__Outer__ is None
        # ) or (
        #         # with magic.default:
        #         #     ...
        #         outer.__outer__ is None
        #         and Default.context
        # ):
        #     # storing in self as self['second'] = second
        #     key = self.__name__
        #     first = magic.__first__
        #     if first is None:
        #         return None
        #     dict = first.__dict__
        if (
                # Magic.magic.second
                outer is None
        ) or (
            # frame.magic.second
            outer.__outer__ is None
            and outer.__Outer__ is None
        ) or (
                # with magic.default:
                #     ...
                outer.__outer__ is None
                and Default.context
        ):
            # storing in self as self['second'] = second
            key = self.__name__
            first = magic.__first__
            if first is None:
                return None
            dict = first.__dict__

        else:
            # storing in outer as outer[self.name] = second
            key = magic.__name__

            if (
                    outer.__first__ is not None
                    and outer.__Outer__ is not None
            ):
                # magic.magic.magic
                dict = outer.__second__.__dict__
            else:
                # magic.magic
                # Magic.magic.magic
                dict = outer.__first__.__dict__

        if key in dict:
            result = dict[key]
        else:
            result = magic.__class__()
            result.__order__ = Order.second
            magic.__first__.__propagate__(result)
            dict[key] = result


        if order == 3:
            cache[name] = weakref.ref(result)

        return result

    def __set__(self, instance: Magic, value):
        if value is None:
            raise ValueError('Cannot set __second__ to None')
        if instance.__order__ != 3:
            raise NotImplementedError
        # if value is not None:
        #     value = weakref.ref(value)
        instance.__cache__[self.__name__] = weakref.ref(value)
        # raise AttributeError('Cannot set __second__')

    def __delete__(self, instance):
        raise AttributeError('Cannot delete __second__')
