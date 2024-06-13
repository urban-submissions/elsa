from __future__ import annotations
from magicpandas.magic.config import Config

import functools
import weakref

import importlib
import inspect
import warnings
from functools import *
from typing import *

from magicpandas.magic.options import Options
from magicpandas.magic.order import Order
from magicpandas.magic.setter import __setter__
from magicpandas.magic.trace import Trace
from magicpandas.magic.default import Default
from typing import TypeVar
T = TypeVar('T')


if False:
    from .magic import Magic


def __get__(self: ABCMagic, instance, owner) -> ABCMagic:
    func = self.__class__.__subget__
    # print(f'{self.__trace__=}')
    result = self.__wrap_descriptor__(func, instance, owner)
    return result


class delayed_import:
    method = None

    @classmethod
    def from_params(cls, func: T) -> T:
        result = delayed_import(func)
        result.method = 'from_params'
        return result

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __set_name__(self, owner, name):
        self.__name__ = name

    def __get__(self, instance, owner):
        name = self.__name__
        *module, cls = (
            inspect.get_annotations(owner)
            [name]
            .rsplit('.', 1)
        )
        if not module:
            module = owner.__module__
        else:
            module = importlib.import_module(module[0])
        if cls == 'Self':
            cls = owner.__name__
        cls = getattr(module, cls)
        if self.method is not None:
            cls = getattr(cls, self.method)
        result = cls(*self.args, **self.kwargs)
        setattr(owner, name, result)
        result.__set_name__(owner, name)
        return result.__get__(instance, owner)


class DelayedImport:
    def __init__(self, func):
        self.func = func

    def __set_name__(self, owner, name):
        self.__name__ = name
        return self

    def __get__(self, instance, owner: type[Magic]):
        attr = self.__name__
        # todo: problem is we're trying to get annotation of child class when
        #   we should get from parent
        annotation = owner.__annotations__[attr]

        if '.' not in annotation:
            name = annotation
            module = inspect.getmodule(owner)
            obj = getattr(module, annotation)
        else:
            module, name = annotation.rsplit('.', 1)
            module = importlib.import_module(module)
            obj = getattr(module, name)
        if (
                obj is Self
                or name == 'Self'
        ):
            obj = owner

        magic = obj(self.func)
        magic.__set_name__(owner, attr)
        setattr(owner, attr, magic)
        result = magic.__get__(instance, owner)
        return result

    def __get__(self, instance, owner: type[Magic]):
        attr = self.__name__
        # todo: problem is we're trying to get annotation of child class when
        #   we should get from parent
        original = owner.__delayed__[attr]
        annotation = original.__annotations__[attr]

        if '.' not in annotation:
            name = annotation
            module = inspect.getmodule(original)
            obj = getattr(module, annotation)
        else:
            module, name = annotation.rsplit('.', 1)
            module = importlib.import_module(module)
            obj = getattr(module, name)
        if (
                obj is Self
                or name == 'Self'
        ):
            obj = original

        magic = obj(self.func)
        magic.__set_name__(original, attr)
        setattr(original, attr, magic)
        result = magic.__get__(instance, owner)
        return result


class ABCMagic:
    """
    mostly has to do with Magic type construction rather than implementation
    """
    __name__: str = ''
    __trace__: Trace
    __order__ = Order.second
    __toggle__: dict[str, bool] = {}
    __setter__ = __setter__()
    locals()['__get__'] = __get__
    __direction__: str
    __options__ = Options()
    __sticky__ = True

    def __subget__(self, instance, owner):
        """Override to handle __get__ for a subclass"""

    def __subset__(self, instance, value):
        """Override to handle __set__ for a subclass"""

    def __subdelete__(self, instance):
        """Override to handle __delete__ for a subclass"""

    def __subinit__(self, *args, **kwargs):
        """Override to handle __init__ for a subclass"""

    def setter(self, func):
        self.__setter__ = func
        return self

    def __set__(self, instance, value):
        func = self.__class__.__subset__
        value = (
            self.__setter__
            .__get__(instance, instance.__class__)
            (value)
        )
        return self.__wrap_descriptor__(func, instance, value)

    def __delete__(self, instance):
        func = self.__class__.__subdelete__
        return self.__wrap_descriptor__(func, instance)

    @cached_property
    def __wrapped__(self):
        """The function that has been wrapped"""

    def __init_func__(self, func, *args, **kwargs):
        raise NotImplementedError

    def __init_nofunc__(self, *args, **kwargs):
        ...

    def __init__(self, *args, **kwargs):
        if (
                args
        ) and (
                inspect.isfunction(args[0])
                or isinstance(args[0], functools.partial)
        ):
            self.__init_nofunc__(*args[1:], **kwargs)
            self.__init_func__(*args, **kwargs)
        else:
            self.__init_nofunc__(*args, **kwargs)

    # third, second, first, root, etc. should be propagated; can this be done without
    #   explicit assignments?
    # noinspection PyMethodParameters

    # todo: get second if outer.__wrapped__

    """
    __max__ = 3
    """

    def __wrap_descriptor__(first: Magic, func, outer: Magic, *args, **kwargs):
        """
        wraps __get__, __set__, __delete__ to handle the "magic" operations
        of propagating all the metadata through the attribute chain
        """

        first.__outer__ = outer
        order = first.__class__.__order__

        if outer is None:
            first.__root__ = root = None
            first.__third__ = None
            first.__owner__ = None
            first.__Root__ = first.__Outer__ = args[0]
        else:
            first.__root__ = root = outer.__root__
            first.__third__ = outer.__third__
            first.__Root__ = outer.__Root__
            if outer is None:
                first.__Outer__ = args[0]
            else:
                first.__Outer__ = type(outer)

        if (
                outer is None
                or order == 1
                # To access nested metadata from class attributes before set_name is called on them
                # or outer.__wrapped__
        ):
            # Frame.magic
            # Frame.frame
            return first

        elif root is None:
            # first.__owner__ = outer.__second__
            try:
                first.__owner__ = outer.__second__
            except AttributeError:
                first.__owner__ = outer.__first__
            match order:
                case 1:
                    return first
                case 3 | 2:
                    # Frame.frame
                    # Frame.frame.magic
                    first.__owner__ = outer.__second__
                    return first.__second__

        else:
            match order:
                case 1:
                    return first
                case 2:
                    # frame.magic.magic
                    # frame.frame.magic

                    # todo problem is we want outer.second to be first.__outer__
                    first.__owner__ = outer.__second__
                    return first.__second__
                case 3:
                    # frame.magic.frame
                    # frame.frame.frame
                    owner = outer.__second__
                    if owner is None:
                        owner = outer.__third__
                    first.__owner__ = owner
                    second = first.__second__
                    inner, outer.__inner__ = outer.__inner__, second
                    second.__owner__ = outer.__third__
                    third: Magic = func(second, outer, *args, **kwargs)
                    outer.__inner__ = inner
                    return third

        return ValueError(f"order {order} not supported")

    """
    how do we get second.edges when second.outer is third order?
    """

    def __wrap_descriptor__(first: Magic, func, outer: Magic, *args, **kwargs):
        """
        wraps __get__, __set__, __delete__ to handle the "magic" operations
        of propagating all the metadata through the attribute chain
        """

        first.__outer__ = outer
        order = first.__class__.__order__
        # todo: problem is this causes config to fail

        if outer is None:
            first.__root__ = root = None
            first.__third__ = None
            first.__owner__ = None
            first.__Root__ = first.__Outer__ = args[0]
        else:
            first.__root__ = root = outer.__root__
            first.__third__ = outer.__third__
            first.__Root__ = outer.__Root__
            if outer is None:
                first.__Outer__ = args[0]
            else:
                first.__Outer__ = type(outer)
        if (
                outer is None
                or order == 1
                # To access nested metadata from class attributes before set_name is called on them
                # or outer.__wrapped__
        ):
            # Frame.magic
            # Frame.frame
            return first

        elif root is None:
            # first.__owner__ = outer.__second__
            try:
                first.__owner__ = outer.__second__
            except AttributeError:
                first.__owner__ = outer.__first__
            match order:
                case 1:
                    return first
                case 3 | 2:
                    # Frame.frame
                    # Frame.frame.magic
                    first.__owner__ = outer.__second__
                    second = first.__second__
                    first.__propagate__(second)
                    return second

        match order:
            case 1:
                return first
            case 2:
                # frame.magic.magic
                # frame.frame.magic
                first.__owner__ = outer.__second__
                second = first.__second__
                first.__propagate__(second)
                return second
            case 3:
                # frame.magic.frame
                # frame.frame.frame
                owner = outer.__second__
                if owner is None:
                    owner = outer.__third__
                first.__owner__ = owner
                second = first.__second__
                inner, outer.__inner__ = outer.__inner__, second
                owner = first.__owner__ = outer.__third__
                # test1 = first.__owner__
                first.__propagate__(second)
                # test2 = second.__owner__
                # second.__owner__ = owner
                # assert second.__owner__ is owner
                # second.__outer__ = outer
                # assert second.__outer__ is outer
                third: Magic = func(second, outer, *args, **kwargs)
                outer.__inner__ = inner
                return third

        raise RuntimeError

    # noinspection PyUnresolvedReferences
    def __init_subclass__(
            cls,
            from_outer=False,
            **kwargs
    ):
        if (
                not from_outer
                and '__from_outer__' in cls.__dict__
        ):
            warnings.warn(f"""
                {cls} defined __from_outer__, which is meant to be used as a
                variable. You most likely intended to define __from_inner__,
                which is the constructor defined for the particular class.

                If you intended to define __from_outer__, you can suppress this 
                warning by setting from_outer=True in the class definition e.g. 
                class MyClass(Magic, from_outer=True):
                    ...
            """, category=UserWarning)

        try:
            from_options = cls.__dict__['from_options']
        except KeyError:
            ...
        else:
            from magicpandas.magic.cached import Base
            try:
                func = from_options.__func__
            except AttributeError as e:
                raise AttributeError(f'{cls.__name__}.from_options must be a classmethod') from e
            kwdefaults = func.__kwdefaults__
            if kwdefaults is None:
                raise NotImplementedError

        # if from_options is defined, assign the defaults
        if 'from_options' in cls.__dict__:
            from_options = cls.__dict__['from_options']
            kwdefaults = cls.from_options.__func__.__kwdefaults__
            for key, value in kwdefaults.items():
                name = f'__{key}__'
                attr = getattr(cls, name)
                try:
                    setattr(cls, name, attr.__ior__(value))
                except (TypeError, AttributeError):
                    ...

            if not isinstance(from_options, classmethod):
                raise ValueError(
                    f"{cls.__module__}.{cls.__name__}.from_options"
                    f" must be a classmethod"
                )

            # noinspection PyUnresolvedReferences
            if from_options.__func__.__code__.co_argcount > 1:
                raise ValueError(f"""
                {cls.__module__}.{cls.__name__}.from_options
                must not have any positional arguments!
                be sure this method looks like this, with the
                'cls', *, and **kwargs:
                @classmethod
                def from_options(cls, *, ..., **kwargs):
                    ...
                """)

        # resolve delayed imports indicated by annotations
        for attr, hint in cls.__annotations__.items():
            # if a method is also annotated as Magic,
            #   it's to be wrapped later.
            # If an ellipsis is annotated as Magic,
            #   it's to be instantiated later.
            attr: str
            try:
                already = getattr(cls, attr)
            except AttributeError:
                continue
            if already is Ellipsis:
                already = None
            elif inspect.isfunction(already):
                ...
            else:
                continue
            delayed = DelayedImport(already)
            delayed.__set_name__(cls, attr)
            setattr(cls, attr, delayed)

        super().__init_subclass__(**kwargs)

    @classmethod
    def from_options(cls, **options) -> Self:
        result = cls()
        if cls.__order__ == 1:
            raise ValueError(f"""
            {cls.from_options} is not a valid option for {cls}
            with order {cls.__order__}
            """)

        for key, value in options.items():
            name = f'__{key}__'
            setattr(result, name, value)

        return result

        # @wraps(cls)
        # def wrapper(*args, **kwargs):
        #     result = cls(*args, **kwargs)
        #     for key, value in options.items():
        #         name = f'__{key}__'
        #         attr = getattr(result, name)
        #         try:
        #             setattr(result, name, attr.__ior__(value))
        #         except (TypeError, AttributeError):
        #             ...
        #     return result
        #
        # return wrapper

import supervision