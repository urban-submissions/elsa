from __future__ import annotations

import argparse
from typing import Callable, TypeVar
from functools import partial
from inspect import Parameter
import inspect
import itertools

from magicpandas.frame import Frame
import collections
from functools import cached_property
import argh
from magicpandas.commandline.arg import Arg, string_type
from weakref import WeakKeyDictionary

"""
@Arg methods under CommandLine only appear when their corresponding commandline is called
"""

if False:
    from magicpandas.root import Root


# todo: magicpandas.commandline, magicpandas.arg, magicpandas.Frame.from_kwargs(align='x y'.split()), type hint magic.Index


class PosArgs:
    cache: dict[Args, PosArgs] = WeakKeyDictionary()

    def __get__(self, instance: Args, owner: type[Args]) -> PosArgs:
        self.owner = instance
        self.Owner = owner
        if instance is None:
            return self
        if instance not in self.cache:
            result = self.__class__()
            object.__setattr__(result, 'owner', instance)
            object.__setattr__(result, 'Owner', owner)
            self.cache[instance] = result
        return self.cache[instance]

    @cached_property
    def order(self) -> list[str]:
        # posargs are sorted in order of build posargs, then call posargs
        args = self.owner
        cmdline = args.owner
        call = cmdline.__call__
        build = cmdline.build
        scall = inspect.signature(call)
        sbuild = inspect.signature(build)
        visited = set()
        result = []
        for name, param in itertools.chain(
                sbuild.parameters.items(),
                scall.parameters.items(),
        ):
            if param.name in visited:
                continue
            if param.default is not Parameter.empty:
                continue
            if param.kind not in {
                Parameter.POSITIONAL_ONLY,
                Parameter.POSITIONAL_OR_KEYWORD,
            }:
                continue

            visited.add(name)
            result.append(name)

        return result

    def add_arguments(self, parser: argparse.ArgumentParser):
        args = self.owner
        scall = inspect.signature(args.owner.__call__)
        sbuild = inspect.signature(args.owner.build)
        # noinspection PyTypeChecker
        parameters = collections.ChainMap(sbuild.parameters, scall.parameters)
        for name in self.order:
            if name in args:
                arg = args[name]
                parser.add_argument(name, **arg)
            else:
                param = parameters[name]
                parser.add_argument(name, type=string_type[param.annotation], default=param.default)


class KWargs:
    cache: dict[Args, KWargs] = WeakKeyDictionary()

    def __get__(self, instance: Args, owner: type[Args]) -> KWargs:
        self.owner = instance
        self.Owner = owner
        if instance is None:
            return self
        if instance not in self.cache:
            result = self.__class__()
            object.__setattr__(result, 'owner', instance)
            object.__setattr__(result, 'Owner', owner)
            self.cache[instance] = result
        return self.cache[instance]

    @cached_property
    def order(self) -> list[str]:
        # kwargs are sorted alphabetically
        args = self.owner
        posargs = set(args.posargs.order)
        result = [
            name
            for name in args
            if name not in posargs
        ]
        result.sort()
        return result

    # todo: how to organize add_argument calls?
    def add_arguments(self, parser: argparse.ArgumentParser):
        for name in self.order:
            arg = self.owner[name]
            parser.add_argument(f'--{name}', **arg)


class Args(collections.UserDict[str, Arg]):
    cache: dict[CommandLine, Args] = WeakKeyDictionary()
    posargs = PosArgs()
    kwargs = KWargs()

    def __get__(self, instance: CommandLine, owner: type[CommandLine]) -> Args:
        self.owner = instance
        self.Owner = owner
        if instance is None:
            return self
        if instance not in self.cache:
            prev = {
                dest: [
                    arg
                    for arg in args
                    if
                    arg.magic.commandline is None
                    or arg.magic.commandline is instance
                ]
                for dest, args in self.owner.Root.magic.commandline.log_args.items()
            }
            result = self.__class__()
            object.__setattr__(result, 'owner', instance)
            object.__setattr__(result, 'Owner', owner)
            self.cache[instance] = result
            i = 1
            while prev:
                # noinspection PyUnusedLocal,PyShadowingBuiltins
                next = collections.defaultdict(list)
                i += 1
                for dest, args in prev.items():
                    if len(args) == 1:
                        # result[args[0]] = dest
                        result[dest] = args[0]
                        continue
                    for arg in args:
                        if i > len(arg):
                            key = dest
                        else:
                            key = arg[i]
                        next[key].append(arg)

                prev = next

        return self.cache[instance]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    T = TypeVar('T', bound=Callable)

    @cached_property
    def order(self) -> list[str]:
        return self.posargs.order + self.kwargs.order

    def preview(self):
        for name in self.posargs.order:
            print(name)
        for name in self.kwargs.order:
            print(f'--{name}')

    def add_arguments(self, parser: argparse.ArgumentParser):
        self.posargs.add_arguments(parser)
        self.kwargs.add_arguments(parser)

    # def dispatch(self):
    #     call = self.owner.__call__
    #     build = self.owner.build
    #
    #     def wrapper(*args, **kwargs):
    #         root: Root = build(*args, **kwargs)
    #
    #     func = self.kwargs.wrap(wrapper)
    #     func = self.posargs.wrap(func)
    #     func.__name__ == self.name
    #     return argh.dispatch_command(func)


class Parser(argparse.ArgumentParser):
    cache: dict[CommandLine, Parser] = WeakKeyDictionary()

    def __get__(self, instance: CommandLine, owner: type[CommandLine]):
        self.owner = instance
        self.Owner = owner
        if instance is None:
            return self
        if instance not in self.cache:
            subparsers = self.owner.Root.magic.commandline.parser.subparsers
            cls, subparsers._parser_class = subparsers._parser_class, self.__class__
            # result = subparsers.add_parser(self.owner.magic.name)
            result = subparsers.add_parser(self.owner.magic.__trace__.rsplit('.', 1)[-1])
            subparsers._parser_class = cls
            self.owner.args.add_arguments(result)
            self.cache[instance] = result
        return self.cache[instance]

    # def __call__(self, *args, **kwargs):
    #     self.parse_args(*args, **kwargs)


class CommandLine(Frame):
    # any Arg nested inside CommandLine will be kept local
    args = Args()
    parser = Parser()

    # todo: But how does Model.train instantiate the model?
    #   probably def construct
    #   or how about __enter__?

    def build(self, *args, **kwargs) -> Root:
        ...

    def __call__(self, *args, **kwargs):
        """
        user-defined
        *args are required and overwrite args if applicable
        train(indir=foobar)

        **kwargs are optional and overwrite args if applicable
        train(lr=.003)

        if args or kwargs are None, default to @Arg value.
        if not None, Arg is returned to default value after call


        Model.train(lr=.002) # use default constructor and call train

        model = Model()

        model.train.lr = .001 # permanently change lr for all calls
        model.train()

        model.train(lr=.002) # temporarily change lr for this call

        python -m train --lr .002   # commandline call with lr=.002

        # todo assure no default value kwargs in param spec
        # this is to assure that the args are being cached instead of passed around

        Model.train() needs filename but not model.train()
        model.train() needs outdir, outdir is not optional so it needs to be included in __call__

        # todo: order is [build.posarg order, call.posarg order, alphabetical kwargs]
        """
        ...
        # todo: assure model has the same metadata as self
        # model = self.root.from_assets(...)

    # @cached_property
    # def wrapped(self):
    #     call = self.__call__
    #     build = self.build
    #
    #     def wrapper(*args, **kwargs):
    #         print(f'args: {args}')
    #         print(f'kwargs: {kwargs}')
    #
    #     func = wrapper
    #     func.__name__ = self.name
    #     func = self.args.kwargs.wrap(wrapper)
    #     func = self.args.posargs.wrap(func)
    #     return func

    # def dispatch(self):
    #     return argh.dispatch_command(self.wrapped)
