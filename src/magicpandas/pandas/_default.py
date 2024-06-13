from __future__ import annotations

import abc
import functools
import inspect
from collections import UserDict
from typing import *

from magicpandas.magic.magic import Magic
from typing import TypeVar
T = TypeVar('T')


class DefaultMethod:
    """Default just updates parameters"""

    def __init__(self, default: default):
        self.default = default
        functools.update_wrapper(self, default.func)

    def __get__(self, instance, owner):
        return functools.partial(self.__call__, instance)

    def passed(self, *args, **kwargs) -> dict[str, Any]:
        default = self.default
        key_meta = default.key_meta
        passed = {
            name: (
                args[i]
                if i < len(args)
                else kwargs[name]
                if name in kwargs
                else key_meta[name]
            )
            for i, (name, param) in enumerate(default.sig.parameters.items())
        }
        return passed

    def update(self, passed: dict[str, Any]):
        default = self.default
        # unpassed defaults are loaded
        update = {}
        for name, value in default.items():
            if not isinstance(value, Magic):
                continue
            trace = str(value.__trace__)
            if passed[name] is None:
                value = value.__from_outer__()
                passed[name] = value
            else:
                value = passed[name]
            update[trace] = value
        return update

    def __call__(self, *args, **kwargs):
        default = self.default
        passed = self.passed(*args, **kwargs)
        self.update(passed)
        result = default.func(**passed)
        return result


class ConstructorMethod(DefaultMethod):
    """Class configures the instance once it has returned."""

    def passed(self, *args, **kwargs) -> dict[str, Any]:
        passed = super().passed(*args, **kwargs)
        key, cls = (
            passed
            .items()
            .__iter__()
            .__next__()
        )
        if not issubclass(cls, Magic):
            raise TypeError(
                f'First argument {key=} must be a Magic subclass; '
                f'got {cls=} instead.'
            )
        return passed

    def __get__(self, instance, owner):
        return functools.partial(self.__call__, owner)

    def __call__(self, *args, **kwargs):
        """
        In default.classmethod, we load the defaults of unpassed params,
        call the method, and configure the defaults after the method
        has returned. This means that the defaults are only available
        outside the method.
        """
        default = self.default
        passed = self.passed(*args, **kwargs)
        update = self.update(passed)
        third: Magic = default.func(**passed)
        with third.configure:
            third.__config__.update(update)
        return third


class ConfigureMethod(DefaultMethod):
    """Bound configures the instance before it is called."""

    def __get__(self, instance, owner):
        return functools.partial(self.__call__, instance)

    def passed(self, *args, **kwargs) -> dict[str, Any]:
        passed = super().passed(*args, **kwargs)
        key, value = (
            passed
            .items()
            .__iter__()
            .__next__()
        )
        if isinstance(value, type):
            raise TypeError(
                f'First argument {key=} must be an instance; got a type. '
                f'To implement a default classmethod, use @default.classmethod.'
            )
        if not isinstance(value, Magic):
            raise TypeError(
                f'First argument {key=} must be a Magic instance; '
                f'got {value=} instead.'
            )
        return passed

    def __call__(self, *args, **kwargs):
        default = self.default
        passed = self.passed(*args, **kwargs)
        update = self.update(passed)
        key, value = (
            passed
            .items()
            .__iter__()
            .__next__()
        )
        third: Magic = value
        with third.configure:
            third.__config__.update(update)
        if third.__inner__:
            result: Magic = default.func(**passed)
            result.__config__ = result.__config__.copy()
        else:
            result = default.func(**passed)
        return result


class method:
    def __init__(self, method):
        self.method = method

    def __get__(self, instance: default, owner) -> type[default] | default:
        if instance is None:
            return self.method
        result = instance.copy()
        result.method = self.method
        return result


class DefaultMeta(abc.ABCMeta):
    def __enter__(self):
        default.context = True
        return default

    def __exit__(self, exc_type, exc_val, exc_tb):
        default.context = False


class default(UserDict, metaclass=DefaultMeta):
    context = False
    method = DefaultMethod
    constructor: method
    configure: method

    def __init__(self, *args, **kwargs):
        # if not self.__class__.context:
        if not default.context:
            raise SyntaxError('The default decorator must be used in a context manager.')
        super().__init__(*args, **kwargs)

    def __call__(self, func: T) -> T:
        self.func = func
        self.sig = inspect.signature(func)
        self.key_meta = {
            name: (
                param.default
                if param.default is not inspect.Parameter.empty
                else None
            )
            for name, param in self.sig.parameters.items()
        }
        default.context, context = True, default.context
        result = self.method(self.copy())
        default.context = context
        return result


class Constructor(default):
    method = ConstructorMethod


class Configure(default):
    method = ConfigureMethod


default.constructor = method(Constructor)
default.configure = method(Configure)


"""
@default.configure(...)
@default(...).configure
@default()

first call returns default copy
second call returns method
"""
