from __future__ import annotations

import functools
import inspect
from collections import UserDict
from functools import *
from typing import *

from magicpandas.magic.config import Config

if False:
    from magicpandas.magic.magic import Magic


class Default(UserDict):
    """
    @magic.default(...)

    @magic.default.configure(...)

    defaults = magic.default(...)
    @defaults
    @defaults.configure
    """
    context = False
    func = None

    def __enter__(self):
        self.__class__.context = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__.context = False

    # @property
    # def configure(self) -> Configure:
    #     result = Configure(self)
    #     result.__dict__.update(self.__dict__)
    #     return result

    @property
    def constructor(self) -> Constructor:
        result = Constructor(self)
        result.__dict__.update(self.__dict__)
        return result

    @cached_property
    def _sig(self):
        return inspect.signature(self.func)

    @cached_property
    def _key_meta(self):
        return {
            name: (
                param.default
                if param.default is not inspect.Parameter.empty
                else None
            )
            for name, param in self._sig.parameters.items()
        }

    def __first__(self, **kwargs) -> Self:
        result = self.__class__(self)
        update = self.__dict__.copy()
        del update['data']
        result.__dict__.update(update)
        result.update(**kwargs)
        return result

    def __second__(self, func) -> Self:
        result = self.__class__(self)
        update = self.__dict__.copy()
        del update['data']
        result.__dict__.update(update)
        result.func = func
        update_wrapper(result, func)
        return result

    def __third__(self, *args, **kwargs):
        passed = self._passed(*args, **kwargs)
        self._passed2config(passed)
        result = self.func(**passed)
        return result

    def __call__(self, *args, **kwargs) -> Self:
        if not self:
            result = self.__first__(**kwargs)
        elif not self.func:
            result = self.__second__(*args, **kwargs)
        else:
            raise NotImplementedError
            result = self.__third__(*args, **kwargs)
        return result

    # todo: config2passed needs to be {instance.trace}.{val.trace}
    def wrapper(self, instance: Magic, owner):
        func = self.func.__get__(instance, owner)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # # obj = owner if instance is None else instance
            passed = self._passed(*args, **kwargs)
            # update = self._config2passed(instance.__config__)
            update = self._config2passed(instance)
            passed.update({
                key: value
                for key, value in update.items()
                if key in passed
                   and passed[key] is None
            })
            result = func(**passed)
            return result

        return wrapper

    def __get__(self, instance, owner):
        if self.func is None:
            return self
        return self.wrapper(instance, owner)

    def __get__(self, instance, owner):
        if self.func is None:
            return self
        if instance is not None:
            return self.wrapper(instance, owner)

        @wraps(self.func)
        def wrapper(instance, *args, **kwargs):
            return self.wrapper(instance, type(instance))(*args, **kwargs)
        return wrapper

    def _passed(self, *args, **kwargs) -> dict[str, Any]:
        key_meta = self._key_meta
        parameters = iter(self._sig.parameters.items())
        next(parameters)
        parameters = enumerate(parameters)
        passed = {
            name: (
                args[i]
                if i < len(args)
                else kwargs[name]
                if name in kwargs
                else key_meta[name]
            )
            for i, (name, param) in parameters
        }
        return passed

    def _passed2config(self, passed: dict[str, Any]) -> dict[str, Any]:
        # unpassed selfs are loaded
        from magicpandas.magic.abc import ABCMagic
        config = {}
        for name, value in self.items():
            if not isinstance(value, ABCMagic):
                continue
            value: Magic
            trace = str(value.__trace__)
            if passed[name] is None:
                value = value.__from_outer__()
                passed[name] = value
            else:
                value = passed[name]
            config[trace] = value
        return config

    # def _config2passed(self, config: dict[str, Any]) -> dict[str, Any]:
    #     passed = {}
    #     from magicpandas.magic.abc import ABCMagic
    #     for name, value in self.items():
    #         if not isinstance(value, ABCMagic):
    #             continue
    #         value: Magic
    #         trace = str(value.__trace__)
    #         if trace in config:
    #             result = config[trace]
    #         else:
    #             result = value.__from_outer__()
    #         passed[name] = result
    #     return passed
    #
    def _config2passed(self, instance:Magic) -> dict[str, Any]:
        passed = {}
        config = instance.__config__
        from magicpandas.magic.abc import ABCMagic
        for name, value in self.items():
            if not isinstance(value, ABCMagic):
                continue
            value: Magic
            trace = f'{instance.__trace__}.{value.__trace__}'
            if trace in config:
                result = config[trace]
            else:
                result = value.__from_outer__()
            passed[name] = result
        return passed

    def __set_name__(self, owner, name):
        self.__name__ = name


class Constructor(Default):
    def __third__(self, *args, **kwargs):
        """
        In default.classmethod, we load the defaults of unpassed params,
        call the method, and configure the defaults after the method
        has returned. This means that the defaults are only available
        outside the method.
        """
        passed = self._passed(*args, **kwargs)
        passed2config = self._passed2config(passed)
        key, value = (
            passed
            .items()
            .__iter__()
            .__next__()
        )

        # todo: also update passed
        from magicpandas.magic.abc import ABCMagic

        if isinstance(value, ABCMagic):
            # We copy the config and update so that the result has an updated
            # config but any updates are not propagated to the original
            value: Magic
            config = value.__config__
            temp = config.copy()
            config.update({
                key: value
                for key, value in passed2config.items()
                if key not in config
                   or config[key] is None
            })
            # config2passed = self._config2passed(config)
            config2passed = self._config2passed(value)
            passed.update({
                key: value
                for key, value in config2passed.items()
                if key in passed
                   and passed[key] is None
            })
            Config.copies, copies = True, Config.copies
            result = self.func(**passed)
            Config.copies = copies
            config.clear()
            config.update(temp)

        elif (
                isinstance(value, type)
                and issubclass(value, ABCMagic)
        ):
            # if we are constructing via classmethod, we configure
            #   the result after it returns
            result: Magic = self.func(**passed)
            result.__config__.update(passed2config)
        else:
            raise TypeError(
                f'First argument {key=} must be a ABCMagic instance; '
                f'got {value=} instead.'
            )

        return result

    def wrapper(self, instance, owner):
        func = self.func.__get__(instance, owner)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # func.__self__
            # instance, owner
            value = owner if instance is None else instance

            passed = self._passed(*args, **kwargs)
            passed2config = self._passed2config(passed)
            # key, value = (
            #     passed
            #     .items()
            #     .__iter__()
            #     .__next__()
            # )

            # todo: also update passed
            from magicpandas.magic.abc import ABCMagic

            if isinstance(value, ABCMagic):
                # We copy the config and update so that the result has an updated
                # config but any updates are not propagated to the original
                value: Magic
                config = value.__config__
                temp = config.copy()
                config.update({
                    key: value
                    for key, value in passed2config.items()
                    if key not in config
                       or config[key] is None
                })
                # config2passed = self._config2passed(config)
                config2passed = self._config2passed(value)

                passed.update({
                    key: value
                    for key, value in config2passed.items()
                    if key in passed
                       and passed[key] is None
                })
                Config.copies, copies = True, Config.copies
                # result = self.func(**passed)
                result = func(**passed)
                Config.copies = copies
                config.clear()
                config.update(temp)

            elif (
                    isinstance(value, type)
                    and issubclass(value, ABCMagic)
            ):
                # if we are constructing via classmethod, we configure
                #   the result after it returns
                result: Magic = self.func(value, **passed)
                result.__config__.update(passed2config)
            else:
                raise TypeError(
                    f'First argument must be a ABCMagic instance; '
                    f'got {value=} instead.'
                )

            return result

        return wrapper



default = Default()
