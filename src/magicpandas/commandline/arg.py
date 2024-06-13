from __future__ import annotations
import collections.abc
import argh

import collections
import inspect
from functools import cached_property
from typing import Type, Optional
from weakref import WeakKeyDictionary
import magicpandas.attr

import magicpandas.magic.cached

# from magicpandas.frame import Frame

if False:
    from magicpandas.commandline.commandline import CommandLine


class StringToType(collections.UserDict):
    def __init__(self):
        super().__init__()
        self.data = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            inspect.Signature.empty: None,
            'None': None,
            None: None,
        }

    def __missing__(self, key):
        raise ValueError(
            f'expected one of {list(self.data.keys())}, got {key}'
        )


string_type = StringToType()


class Arg(magicpandas.attr, collections.abc.Mapping):
    def keys(self):
        return 'type default help'.split()

    # def __getitem__(self, item):
    #     return getattr(self, item)

    def __iter__(self):
        yield from self.keys()

    owner: Frame
    Owner: Type[Frame]

    @cached_property
    def help(self):
        return self.fget.__doc__

    # @cached_property
    # def type(self):
    #     # inspect.signature(self.fget).return_annotation
    #     # return inspect.signature(self.fget).return_annotation
    #     signature = inspect.signature(self.fget)
    #     if signature.return_annotation is not inspect.Signature.empty:
    #         match signature.return_annotation:
    #             case 'str':
    #                 return str
    #             case 'int':
    #                 return int
    #             case 'float':
    #                 return float
    #             case 'bool':
    #                 return bool
    #             case _:
    #                 raise NotImplementedError(
    #                     f'{self.traceback} received non-elementary type {signature.return_annotation}'
    #                 )
    #     default = self.default
    #     if default is None:
    #         return None
    #     return type(default)

    # if signature.return_annotation is inspect.Signature.empty:
    #     return type()

    @cached_property
    def type(self):
        key = inspect.signature(self.fget).return_annotation
        return string_type[key]

    @cached_property
    def default(self):
        return self.fget(self.owner)

    @cached_property
    def traceback(self) -> str:
        try:
            return self.owner.__trace__ + '.' + self.trace
        except (AttributeError, ValueError):
            return self.trace

    @cached_property
    def split(self):
        return self.traceback.split('.')

    @cached_property
    def dest(self) -> str:
        return magicpandas.magic.cached.Root.log_args.arg_dest[self]

    @cached_property
    def commandline(self) -> Optional[CommandLine]:
        from magicpandas.commandline.commandline import CommandLine
        owner = self
        while hasattr(owner, 'owner'):
            if isinstance(owner, CommandLine):
                return owner
            owner = owner.owner

    # @cached_property
    # def wrap(self):
    #     # todo: store_true and store_false
    #     # path = self.traceback
    #     # build = self.commandline.build
    #     result = argh.arg(
    #         # self.traceback,
    #         help=self.help,
    #         type=self.type,
    #         default=self.default,
    #     )
    #     return result
    #

    def __hash__(self):
        return hash(self.traceback)

    def __eq__(self, other):
        return self.traceback == other

    def __getitem__(self, item):
        if isinstance(item, int):
            return '.'.join(self.split[-item:])
        if isinstance(item, str):
            return getattr(self, item)
        raise TypeError(f'{item} must be int or str')
        # return '.'.join(self.split[-item:])

    def __len__(self):
        return len(self.split)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.traceback})'

    # todo: cache needs to be specific to instance
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # noinspection PyTypeChecker
        self.cache: dict[Frame, Arg] = WeakKeyDictionary()

    def __get__(self, instance: Optional[Frame], owner: Type[Frame]):
        # each Arg is unique to the instance
        if instance is None:
            return super().__get__(instance, owner)
        if instance not in self.cache:
            # unique per instance
            result = self.__class__(self.fget)
            object.__setattr__(result, 'owner', instance)
            object.__setattr__(result, 'Owner', owner)
            object.__setattr__(result, 'name', self.trace)
            self.cache[instance] = result
        result = self.cache[instance]
        if magicpandas.magic.cached.Root is None:
            # if root is None, return the arg
            return result
        return super(self.__class__, result).__get__(instance, owner)


arg = Arg
