from __future__ import annotations

import contextlib
import math
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from functools import lru_cache
from typing import *
from typing import Self

from magicpandas.magic.delayed import Delayed
from magicpandas.logger.logger import ConsoleLogger
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.propagating import Propagating
from magicpandas.magic.cached import cached
from magicpandas.magic.config import Config
from magicpandas.magic.directions import Directions, Direction
from magicpandas.magic.from_params import FromParams
from magicpandas.magic.options import Options
from magicpandas.magic.order import Order
from magicpandas.magic.root import Root
from magicpandas.magic.second import Second
from magicpandas.magic.third import Third
from magicpandas.magic.trace import Trace
from magicpandas.magic.truthy import truthy
from magicpandas.magic.volatile import Volatile
from magicpandas.magic.stickies import Stickies
from functools import cached_property
import math
from typing import TypeVar
T = TypeVar('T')

if False:
    from ..pandas.ndframe import NDFrame


class Magic(
    ABCMagic,
    from_outer=True
):
    __direction__: str | Direction = Directions.horizontal
    __directions__ = Directions()
    __second__: Self = Second()
    __third__ = Third()
    __root__ = Root()
    __volatile__ = Volatile()
    __config__ = Config()
    __from_params__ = FromParams()
    __options__ = Options()
    __sticky__ = True
    __stickies__ = Stickies()
    __propagating__ = Propagating()
    __trace__ = Trace()
    __delayed__ = Delayed()

    def __subget__(self: Magic, outer: Magic, Outer: type[Magic]) -> Magic:
        """ always runs when accessing 2nd order """
        if outer.__root__.__configuring__:
            return self
        owner = outer.__getattribute__(Order.second)
        key = self.__key__
        if key not in owner.__dict__:
            self.__subset__(outer, self)
        return owner.__dict__[key]

    def __subset__(self, outer: Magic, value):
        owner = self.__owner__
        key = self.__key__
        self.__propagate__(value)
        owner.__dict__[key] = value
        return value

    def __subdelete__(self, outer: Magic):
        owner = self.__owner__
        key = self.__key__
        try:
            del owner.__dict__[key]
        except KeyError:
            ...

    def __propagate__(self, obj: Magic | T) -> T:
        """ set metadata from another object """
        if obj is None:
            return
        if self.__order__ == 2:
            _ = self.__second__
        if obj.__order__ == self.__order__:
            cache = self.__directions__.horizontal
        elif obj.__order__ > self.__order__:
            cache = self.__directions__.vertical
        else:
            raise ValueError(f'obj.order < self.order')
        diagonal = self.__directions__.diagonal
        obj.__dict__.update(
            (key, value)
            for key, value in self.__dict__.items()
            if key in cache
            or key in diagonal
        )
        obj.__volatile__.update(self.__volatile__)

        return obj

    def __set_name__(self, Outer, name):
        self.__name__ = name
        self.__first__ = self
        self.__order__ = Order.first
        self.__Outer__ = Outer
        self.__third__ = None

        # del self.__trace__
        # _ = self.__trace__

        # self.__trace__.retrace()
        # # recompute trace
        # for key, value in self.__dict__.items():
        #     if not isinstance(value, Magic):
        #         continue
        #     second = getattr(self, key)
        #     second.__trace__.retrace()

    # @lru_cache()
    def __repr__(self):
        try:
            result = self.__trace__.__str__()
            match self.__order__:
                case Order.first:
                    result += ' 1st'
                case Order.second:
                    result += ' 2nd'
                case Order.third:
                    result += ' 3rd'

            return result
        except AttributeError:
            return super().__repr__()

    @cached_property
    def __directory__(self):
        """
        the file path of the current instance; this is used for caching
        """
        # cwd/magic/magic
        return os.path.join(
            self.__rootdir__,
            self.__trace__.replace('.', os.sep),
        )

    @cached.root.property
    def __timeit__(self) -> float:
        """
        If a subprocess took more than this long to complete,
        log the time taken. If 0, every time is logged.
        If -1, no time is logged. This can be overridden with
        @cached.property so that the time is specific to instance.
        """
        return 1.

    @property
    def __cache__(self):
        return self.__dict__

    @cached.volatile.property
    def __outer__(self) -> Self | ABCMagic | NDFrame:
        """
        The object immediately outside this nested object;

        class Outer(magic):
            @Inner
            def inner(self):
                ...

        outer = Outer()
        outer.inner

        Here, inner is nested in outer
        """
        return

    @cached.volatile.property
    def __owner__(self) -> Self | ABCMagic | NDFrame:
        """
        The object that the __dict__ containing this attribute is attached to;

        class Outer(magic):
            @Inner
            def inner(self):
                ...

        class Owner(frame):
            outer = Outer()

        owner = Owner()
        owner.outer.inner

        Here, inner is owned by owner;
        if you look in owner.__dict__ you will find 'outer.inner'
        """
        return

    @cached.volatile.property
    def __Outer__(self) -> type[Self]:
        """
        The type of the outer class;

        class Outer(magic):
            @Inner
            def inner(self):
                ...
        Outer.inner

        Here, inner.__outer__ is None, but inner.__Outer__ is Outer
        """
        return

    @cached.volatile.property
    def __inner__(self) -> Self | ABCMagic | NDFrame:
        """
        The instance of the object for which the current method is being called;

        class Outer(magic):
            @Inner
            def inner(self: Outer):
                self.__inner__: Outer
                return self.__inner__({
                    'a': [1,2,3],
                })

        Here, we have a simple method with the typical bound self,
        but for some reason we may need to access metadata about the
        inner instance to be constructed. For that purpose we have self.__inner__
        """
        return

    #
    # @cached.volatile.property
    # def __max__(self) -> int:
    #     order = self.__order__.__int__()
    #     outer = self.__outer__
    #     if outer is None:
    #         return order
    #     elif int(outer.__order__) > order:
    #         return outer.__max__
    #     elif order != self.__class__.__order__:
    #         return min(outer.__max__, order)
    #     return outer.__max__
    #
    # # frame.second.ow
    #
    # """
    # todo: we need to set first with max in every wrap_descriptoe
    #     and propagate from first to second
    # """
    #
    # @cached.volatile.property
    # def __max__(self) -> int:
    #     # Magic.magic 2
    #     # magic.magic
    #     # Magic.magic.frame 2
    #     # magic.frame.second 2
    #     # magic.frame.second.frmae 2
    #     # frame.frame.frame
    #     # frame.frame.magic
    #
    #     order = self.__order__.__int__()
    #     outer = self.__outer__
    #     if order == self.__class__.__order__:
    #         order = math.inf
    #     if outer is None:
    #         return order
    #     return min(order, outer.__max__)
    #

    @cached.property
    def __max__(self) -> float:
        order = self.__order__
        if order != self.__class__.__order__:
            this = int(order)
        else:
            this = 3
        outer: Magic = self.__outer__
        if outer is None:
            prev = 3
        else:
            prev = outer.__max__
        result = min((this, prev))
        return result





    # @cached.property
    @cached.diagonal.property
    def __name__(self) -> str:
        """
        The name of the attribute in the access chain;
        For first.second.third, the __name__ values are
        '', 'second', 'third', respectively; the root has no name.
        """
        if self.__outer__ is None:
            return ''
        return self.__first__.__name__

    @cached.volatile.property
    def __Root__(self) -> type[Self]:
        """
        Root.frame.magic.__Root__ is Root
        """
        return

    # @cached.property
    @cached.diagonal.property
    def __first__(self) -> Self | Any | ABCMagic | NDFrame:
        """
        Commutative metadata
        a.b is c.b

        The first order instance of the hierarchy

        class Outer(frame):
            inner = Inner()

        Here, the object literally created by the line
        `inner = Inner()` is the first order instance,
        and is stored in Outer.__dict__['inner']

        The first order instance only contains metadata,
        and does not contain data associated with the process.

        First.second:
        name:           First   second
        cls.order:      3       2
        instance.order: -       1

        """
        if self.__class__.__order__ == 1:
            return self

    @cached.root.property
    def __futures__(self) -> list[Future]:
        """
        The futures that are currently running for this process;
        all futures are awaiting before the process ends.
        """
        return []

    @cached.root.property
    def __threads__(self) -> ThreadPoolExecutor:
        """
        The thread pool associated with this process
        """
        return ThreadPoolExecutor()

    @cached.root.property
    def __logger__(self) -> ConsoleLogger:
        from magicpandas.logger.logger import logger
        return logger

    @cached.root.property
    def __done__(self) -> bool:
        """
        Whether the process is done;
        this is used to determine whether to await the process
        """
        return True

    @cached.root.property
    def __rootdir__(self) -> str:
        # Magic.__init__
        """
        root directory of specific instance;
        all cached files are stored in subdirectories
        """
        # tempfile.tempdir
        # return os.getcwd()

    @cached.root.property
    def __configuring__(self) -> bool:
        return False

    @cached.property
    def __calling_from_params__(self) -> bool:
        return bool(self.__from_params__)

    @property
    def __key__(self) -> str:
        """
        The key for the current instance; this is used for caching
        Has to be string or else pandas.NDFrmae.__getitem__ will fail
        because Trace is a callable.
        """
        trace = self.__trace__
        owner = self.__owner__
        if owner is None:
            return str(trace - self.__outer__.__trace__)
        return str(trace - owner.__trace__)

    @truthy
    def __from_outer__(self):
        """
        the function belonging to outer that will generate the attribute

        class Outer(Frame):

            @Inner
            def inner(self: Outer) -> Inner:
                return self.__inner__(self)

        here, def thing becomes thing.__from_outer__, because self is outer
        Regardless, it returns an instance of inner
        """
        raise NotImplementedError(
            f"No constructor has been defined for {self.__trace__}. If the "
            f"method is intended to return None,"
        )

    @truthy
    def __postprocess__(self, result: T) -> T:
        return result

    @property
    def configure(self):
        @contextlib.contextmanager
        def configure():
            root = self.__root__
            config = root.__configuring__
            root.__configuring__ = True
            yield self
            root.__configuring__ = config

        return configure()

    @property
    def freeze(self):
        raise NotImplementedError

        @contextlib.contextmanager
        def freeze():
            yield self

        return freeze()

    def __getnested__(self, key: str) -> Magic:
        """
        get a nested object from the current object
        """
        obj = self
        # column access adds to permanent
        for piece in key.split('.'):
            obj = getattr(obj, piece)
        return obj


if __name__ == '__main__':
    assert Trace('a.b.c') - 'a' == 'b.c'
    assert Trace('a.b.c') - 'a.b' == 'c'

if __name__ == '__main__':
    class Fourth(Magic):
        first = Magic()
        second = Magic()


    class Third(Magic):
        first = Fourth()
        second = Fourth()


    class Second(Magic):
        first = Third()
        second = Third()


    class First(Magic):
        first = Second()
        second = Second()


    print(f'{First.first.__trace__=}')
    second = First.first.second
    print(f'{second.__trace__=}')
    print(f'{First.second.first.second.__trace__=}')
    print(f'{First.second.first.second.first.__trace__=}')
    print(f'{First.second.first.first.__trace__=}')

    assert First.first.__trace__ == 'first'
    assert First.first.first.__trace__ == 'first.first'
    assert First.second.first.second.__trace__ == 'second.first.second'
    assert First.second.first.second.first.__trace__ == 'second.first.second.first'
    assert First.second.first.first.__trace__ == 'second.first.first'
    from magicpandas.pandas.ndframe import NDFrame


