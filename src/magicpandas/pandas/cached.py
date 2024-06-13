from __future__ import annotations

import contextlib
import functools
import inspect
import os.path
import pickle
import tempfile
import time
import warnings
import weakref
from functools import *
from functools import cached_property
from functools import lru_cache
from types import FunctionType
from typing import *

from magicpandas.magic.default import default
import magicpandas as magic
import magicpandas.magic.util as util
from magicpandas.magic import cached as _cached
from magicpandas.magic import magic
from magicpandas.magic.abc import ABCMagic
from magicpandas.magic.cached import Diagonal
from magicpandas.magic.cached import Volatile
from magicpandas.magic.truthy import truthy
from typing import TypeVar

T = TypeVar('T')

if False:
    from .ndframe import NDFrame


def __get__(self, instance: NDFrame, owner):
    if instance is None:
        return self
    key = self.__name__
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
    return result


class Base(_cached.Base):
    """Base does not use outer, owner, etc."""
    locals()['__get__'] = __get__

    def __init__(self, __func__: FunctionType | Magic, *args, **kwargs):
        if isinstance(__func__, _cached.Base):
            self.__dict__.update(__func__.__dict__)
        else:
            super().__init__(__func__, *args, **kwargs)

    def __set__(self, instance: NDFrame, value):
        if isinstance(value, ABCMagic):
            value = weakref.ref(value)
        if self.__setter__ is not None:
            value = self.__setter__(value)
        # instance.attrs[self.__name__] = value
        instance.__cache__[self.__name__] = value

    def __delete__(self, instance: NDFrame):
        try:
            del instance.__cache__[self.__name__]
        except KeyError:
            ...

    @cached_property
    def __setter__(self) -> Callable[[T], T] | None:
        ...

    def setter(self, func):
        self.__setter__ = func
        return self


class Frame(Base):
    def __set__(self, instance: NDFrame, value):
        if self.__setter__ is not None:
            value = self.__setter__(value)
        instance.__cache__[self.__name__] = value


class Magic(magic.Magic, _cached.Base):
    """ Extends magic.cached.property to use pandas.NDFrame.attrs """
    __outer__: NDFrame
    __owner__: NDFrame
    __order__ = magic.Magic.__order__.third
    __from_inner__ = None

    def __set_name__(self, owner: magic.Magic, name):
        magic.Magic.__set_name__(self, owner, name)
        _cached.Base.__set_name__(self, owner, name)

    def __init_func__(self, func=None, *args, **kwargs):
        if isinstance(func, Base):
            self.__dict__.update(func.__dict__)
        else:
            super().__subinit__(func, *args, **kwargs)

        # assert inspect.isfunction(func)
        parameters = inspect.signature(func).parameters
        functools.update_wrapper(self, func)

        # if not (
        #     util.returns(func)
        #     or util.contains_functioning_code(func)
        # ):
        #     if not self.__from_inner__:
        #         self.__permanent__ = True
        # elif len(parameters) > 1:
        #     self.__from_params__ = func
        # else:
        #     self.__from_outer__ = func

        if len(parameters) > 1:
            # case from params
            self.__from_params__ = func
        elif (
                not util.returns(func)
                and not util.contains_functioning_code(func)
        ):
            if not self.__from_inner__:
                self.__permanent__ = True
        else:
            self.__from_outer__ = func

        # elif (
        #     not util.returns(func)
        #     and not util.contains_functioning_code(func)
        # ):
        #     self.__permanent__ = True
        # else:
        #     self.__from_outer__ = func

    def __subget__(self, outer: Magic, Outer):
        owner: NDFrame
        if outer is None:
            return self

        elif default.context:
            return self

        elif self.__configuring__:
            key = self.__trace__.__str__()
            if key not in self.__config__:
                self.__config__[key] = self.__from_outer__()
            result = self.__config__[key]
            return result

        elif (
                outer is not None
                and outer.__max__ < 3
        ):
            return self

        # noinspection PyTypeChecker
        owner = self.__owner__
        key = self.__key__
        trace = self.__trace__.__str__()
        # if key in owner.attrs:
        #     # get from cached instance attr
        #     result = owner.attrs[key]
        #     if isinstance(result, weakref.ref):
        #         result = result()
        #     return result
        if key in owner.__cache__:
            # get from cached instance attr
            result = owner.__cache__[key]
            if isinstance(result, weakref.ref):
                result = result()
            return result

        if trace in owner.__config__:
            # get from config
            return owner.__config__[trace]

        # todo: maybe use volatile in wrap_descriptor instead to minimize user error
        volatile = self.__volatile__.copy()

        if (
                self.__from_file__
                and os.path.exists(self)
        ):
            # load from file
            result = self.__from_file__()
            # todo: could this cause a memory leak?
            try:
                result.__unlink__ = self.__unlink__
            except AttributeError as e:
                warnings.warn(str(e))

        elif self.__from_outer__:
            # compute from func
            func = self.__from_outer__.__func__.__get__(outer, type(outer))
            outer.__inner__, inner = self, outer.__inner__
            if self.__log__:
                result = self.__log__(func)
            else:
                result = func()
            outer.__inner__ = inner

        elif self.__from_inner__:
            # load from inner
            if self.__log__:
                result = self.__log__(self.__from_inner__, outer)
            else:
                result = self.__from_inner__()
        else:
            raise ValueError(
                f'Could not resolve a constructor for {self.__trace__}. '
                f'If get-before-set is acceptable, you must explicitly return None.'
            )

        # noinspection PyUnresolvedReferences
        self.__subset__(outer, result)
        # result = owner.attrs[key]
        result = owner.__cache__[key]
        if isinstance(result, weakref.ref):
            result = result()
        if (
                self.__from_file__
                and not os.path.exists(self)
        ):
            self.__to_file__(result)

        self.__volatile__.update(volatile)

        return result

    def __subset__(self, instance: NDFrame, value):
        if isinstance(value, ABCMagic):
            value = weakref.ref(value)

        if self.__configuring__:
            cache = self.__config__
            key = self.__trace__.__str__()
        else:
            # cache = self.__owner__.attrs
            cache = self.__owner__.__cache__
            key = self.__key__
        cache[key] = value

        return value

    def __subdelete__(self, instance: NDFrame):
        if self.__configuring__:
            cache = self.__config__
            key = self.__trace__.__str__()
        else:
            cache = self.__owner__.__cache__
            key = self.__key__
        if key in cache:
            del cache[key]

    @_cached.cached.property
    def __postinit__(self):
        """
        If True, the column will be initialized after the initialization
        of the owner, rather than needing to be accessed first.
        """
        return False

    # @_cached.cached.property
    @_cached.cached.root.property
    def __rootfile__(self) -> Optional[str]:
        return None

    @_cached.cached.root.property
    def __rootdir__(self) -> str:
        rootfile = self.__rootfile__
        if rootfile is None:
            raise AttributeError(f'{self.__trace__}.__rootfile__ is not set')
        filename = (
            self.__rootfile__
            .rsplit(os.sep, 1)[-1]
            .split('.')[0]
        )
        dir = tempfile.tempdir
        module = self.__class__.__module__
        result = os.path.join(dir, module, filename)
        return result

    # @_cached.cached.property
    @_cached.cached.diagonal.property
    def __no_recursion__(self) -> bool:
        """If True, raises AttributeError if recursively accessed."""
        return False

    @_cached.cached.property
    def __is_recursion__(self) -> bool:
        """If True, the cached property is already being accessed."""
        return False

    @cached_property
    def __recursions__(self) -> set[str]:
        return set()

    @contextlib.contextmanager
    def __recursion__(self):
        cache = self.__owner__.__recursions__
        key = self.__key__
        if (
                self.__no_recursion__
                and cache.setdefault(key, False)
        ):
            raise RecursionError(
                f'{self.__trace__} is recursively defined.'
            )
        cache[key] = True
        try:
            yield
        except Exception as e:
            cache.clear()
            raise e
        else:
            cache.clear()

    @contextlib.contextmanager
    def __recursion__(self):
        cache = self.__owner__.__recursions__
        key = self.__key__
        if (
                self.__no_recursion__
                and key in cache
        ):
            raise RecursionError(
                f'{self.__trace__} is recursively defined.'
            )
        empty = not cache
        cache.add(key)
        try:
            yield
        except Exception as e:
            cache.remove(key)
            raise e
        else:
            cache.remove(key)
            if empty:
                cache.clear()

    @classmethod
    def __from_pipe__(cls, *args, **kwargs) -> Self:
        """
        # todo: create this option from commandline pipe e.g.
        <other process> | python <project> first.second.third
        """
        raise NotImplementedError

    @classmethod
    def __from_commandline__(cls, *args, **kwargs) -> Self:
        """
        in commandline:
        python <project> first.second.third
        """
        raise NotImplementedError

    @truthy
    def __from_file__(self):
        with open(self, 'rb') as file:
            result: Self = pickle.load(file)

    def __to_file__(self, value=None):
        if value is None:
            value = self

        def serialize():
            with open(self, 'wb') as file:
                pickle.dump(value, file)

        future = self.__root__.__threads__.submit(serialize)
        self.__root__.__futures__.append(future)

    def __unlink__(self):
        os.unlink(self)

    def __fspath__(self):
        # cwd/magic/magic.pkl
        return self.__directory__ + '.pkl'

    @truthy
    def __log__(
            self,
            func: FunctionType | Any,
            *args,
            **kwargs,
    ):
        """
        log information about the current subprocess
        """
        # todo: allow user to change log level
        if not self.__log__:
            return func(*args, **kwargs)
        logger = self.__logger__
        T = self.__timeit__
        t = time.time()
        # logger.info(f'{self.__trace__}.{func.__name__}')
        logger.info(self.__trace__)
        logger.indent += 1
        result = func(*args, **kwargs)
        t = time.time() - t
        if (
                T is not None
                and 0 <= T <= t
        ):
            logger.info(f'{t=:.2f}s')

        logger.indent -= 1
        return result

    @classmethod
    def from_options(
            cls,
            *,
            postinit=False,
            log=True,
            from_file=False,
            no_recursion=False,
            **options
    ) -> Callable[[...], Self]:
        return super().from_options(
            postinit=postinit,
            log=log,
            from_file=from_file,
            no_recursion=no_recursion,
            **options
        )


class Root(Magic):
    __subset__ = Magic.__subset__
    __subdelete__ = Magic.__subdelete__

    # noinspection PyUnresolvedReferences,PyRedeclaration
    def __subget__(self: Root, instance: Magic, owner):
        if instance is None:
            return self
        if default.context:
            return self
        if instance.__root__ is not None:
            instance = instance.__root__
        return super().__subget__(instance, owner)

    def __wrap_descriptor__(first, func, outer, *args, **kwargs):
        if (
                outer is not None
                and outer.__root__ is not None
        ):
            outer = outer.__root__
        return super().__wrap_descriptor__(func, outer, *args, **kwargs)


class cached:
    property = Base

    class base:
        property = Base

    class magic:
        property = Magic

    class root:
        property = Root

    class cmdline:
        property = Magic

    class volatile:
        property = Volatile

    class diagonal:
        property = Diagonal

    class frame:
        property = Frame
