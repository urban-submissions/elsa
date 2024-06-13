from __future__ import annotations

from datetime import datetime
from typing import Type

import numpy as np
from pandas import Series, Index

import magicpandas as magic
import magicpandas.pandas.frame
import magicpandas.pandas.index


class Value(magicpandas.pandas.frame.Frame):
    # magic.Index() allows for an index to be accessed as an attribute
    # agnostic to whether it is currently an Index, MultiIndex, or Series

    iapple = magicpandas.pandas.index.Index()

    # type-hint columns to allow for type-hinted, legible, and autocompleted column access
    # here we have just a regular column, not a magic.column; this is defined at construction
    # and not lazily/dynamically evaluated

    usd: Series[float]

    # use @magic.column decorator for magic columns, which are a pythonic way to define columns
    # magic.column allows for a column to defined with a method to be lazy evaluated

    @magic.column
    def ruble(self) -> Series[float]:
        result = self.usd * 100000
        return result

    # magic.attr allows for an attribute to be defined with a method to be lazy evaluated

    # @magic.attr
    # @property
    # def magic.timestamp(self) -> datetime:
    #     return datetime.now()

    # use the ellipsis as a placeholder for incomplete code or lines where the value or operation does not matter

    @magic.attr
    def rmb(self) -> Series[float]:
        ...


class Juice(magicpandas.pandas.frame.Frame):
    owner: Apple
    iapple = magicpandas.pandas.index.Index()

    # from_self returns an instance of self and is defined in the same class it returns
    # each class can only have one definition of from_self;
    # this is the standard way of defining a magic.Frame constructor

    @magic.console.wrap.debug
    def from_inner(self, owner: Apple) -> Juice:
        index = owner.index
        result = self.__class__({}, index=index)
        return result

    # magic.console.wrap.debug will wrap the magic column method to be logged to the console
    # even though this method returns a numpy array, the wrapper ensures a Series is returned

    @magic.column
    @magic.console.wrap.debug
    def volume(self):
        result = (
                self.owner
                .loc[self.index]
                .volume.values
                / 3
        )
        return result

    # from_owner returns an instance of the class from which the from_owner method is called
    # each clas can have multiple definitions using from_owner;
    # this constructor allows for reusing the same class in different contexts

    @Value.from_outer
    @magic.console.wrap.debug
    def value(self, owner: Value) -> Value:
        usd = self.volume * 3
        result = owner.__class__({
            'usd': usd
        })
        return result


class Vinegar(magicpandas.pandas.frame.Frame):
    owner: Cider
    iapple = magicpandas.pandas.index.Index()

    @magic.console.wrap.debug
    def from_inner(self, owner: Cider) -> Vinegar:
        index = owner.index
        result = self.__class__({}, index=index)
        return result

    @magic.column
    @magic.console.wrap.debug
    def volume(self):
        cider: Cider = self.owner.loc[self.index]
        result = cider.volume / 2
        return result

    # Value.from_owner constructs the magic frame Value from the owner of the magic frame Vinegar

    @Value.from_outer
    @magic.console.wrap.debug
    def value(self, owner: Value) -> Value:
        usd = self.volume * 3
        result = owner.__class__({
            'usd': usd
        })
        return result

    @magic.column
    def ph(self) -> Series[float]:
        # to be implemented later, for now just leave an ellipsis
        ...


class Cider(magicpandas.pandas.frame.Frame):
    owner: Apple
    vinegar = Vinegar()

    # magic.Frames can be nested into other magic.Frames() simply by instantiating an empty instancee
    # in the owner class; when apple.cider.vinegar is accessed, a new instance of Vinegar is created
    # using Vinegar.from_self(owner=apple.cider)

    @magic.console.wrap.debug
    def from_inner(self, owner: Apple) -> Cider:
        index = owner.index
        result = self.__class__({}, index=index)
        return result

    @Value.from_outer
    @magic.console.wrap.debug
    def value(self, owner: Value) -> Value:
        usd = self.owner.value.usd * 5
        result = owner.__class__({
            'usd': usd
        })
        return result

    @magic.column
    @magic.console.wrap.debug
    def volume(self):
        apples: Apple = self.owner.loc[self.index]
        result = apples.volume / 2
        return result


class Supply(magicpandas.pandas.frame.Frame):
    owner: Market
    commodity: magicpandas.pandas.index.Index()
    value: Series[float]

    @magic.console.wrap.debug
    def from_inner(self, owner: Market) -> Supply:
        password = self.owner.password
        username = self.owner.username
        magic.logger.console.info(f'logging in with {username} and {password}')
        commodity = 'apple juice cider vinegar'.split()
        value = [10000, 1000, 100, 10]
        index = Index(commodity, name='commodity')
        result = self.__class__({
            'value': value
        }, index=index)
        return result


class Demand(magicpandas.pandas.frame.Frame):
    commodity = magicpandas.pandas.index.Index()
    owner: Market
    good: Series[str]
    value: Series[float]

    @magic.console.wrap.debug
    def from_inner(self, owner: Market) -> Demand:
        password = self.owner.password
        username = self.owner.username
        magic.logger.console.info(f'logging in with {username} and {password}')
        commodity = 'apple juice cider vinegar'.split()
        index = Index(commodity, name='commodity')
        value = [10000, 1000, 100, 10]
        result = self.__class__({
            'value': value
        }, index=index)
        return result


class Market(magicpandas.pandas.frame.Frame):
    # frames can be defined without descriptors and used just for structural organization;
    # here we have apple.market.supply and apple.market.demand
    # where apple.market is syntactic sugar but also contains metadata
    commodity = magicpandas.pandas.index.Index()
    owner: Apple
    supply = Supply()
    demand = Demand()

    @magic.attr
    @magic.console.wrap.debug
    def password(self):
        return 'password'

    @magic.attr
    def username(self):
        return 'username'


class Apple(magic.Root):
    # magic.Root is the outermost frame in the implemented hierarchy, creating an interface such as
    # apple.cider.vinegar; the root typically represents the source data from which the rest of the
    # process is derived

    # apple contains nested magic frames cider, juice, and market, constructed with magic.Frame.from_self

    iapple = magicpandas.pandas.index.Index()
    cider = Cider()
    juice = Juice()
    market = Market()

    @classmethod
    def from_random(cls, n: int) -> Apple:
        index = Index(range(n), name='iapple')
        result = cls(index=index)
        return result

    # magic.column init=True forces the column to be evaluated at construction
    @magic.column(init=True)
    @magic.console.wrap.debug
    def volume(self) -> Series[float]:
        mean = 100
        std = 10
        result = np.random.normal(mean, std, len(self))
        return result

    @magic.column
    @magic.console.wrap.debug
    def mass(self) -> Series[float]:
        # linear transformation based on correlations
        volume = self.volume.values
        mass = volume * 160 / 100
        # todo: what is correct std scaling?
        std = volume * 20 / 160
        random = np.random.normal(0, std, len(self))
        result = mass + random
        return result

    # magic.Frame.from_owner allows the construction to be defined in the owner, Apple,
    # rather than the nested frame, Value; this allows for the nested magic.Frame to be created in different
    # ways without redefining the class

    @Value.from_outer
    @magic.console.wrap.debug
    def value(self, owner: Value) -> Value:
        dollars = self.mass * 3 / 100
        result = owner.__class__({
            'usd': dollars
        })
        return result


if __name__ == '__main__':
    # apple = Apple.from_random(10)
    apple = Apple.from_random(10)
    apple.juice
    juice = apple.juice
    apple.volume
    juice.volume
    apple.juice.value
    apple.cider
    apple.cider.value.usd
    apple.cider.vinegar
    apple.cider.vinegar.value.ruble


class Juice(magicpandas.pandas.frame.Frame):
    owner: Apple

    def from_inner(self, owner: Apple) -> Juice:
        ...


class Apple(magicpandas.pandas.frame.Frame):
    juice = Juice()


class Juice(magicpandas.pandas.frame.Frame):
    owner: Apple


class Apple(magicpandas.pandas.frame.Frame):
    @Juice.from_outer
    def juice(self, meta: Juice) -> Juice:
        ...
