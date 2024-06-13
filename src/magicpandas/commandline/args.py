from __future__ import annotations

import collections
import inspect
import json
import pprint
from functools import cached_property
from typing import Type, Optional, Iterator
from weakref import WeakKeyDictionary

import argh
import pandas as pd

import magicpandas.attr
from magicpandas.frame import Frame
from magicpandas.commandline.arg import Arg
from magicpandas.commandline.commandline import CommandLine

if False:
    from magicpandas.root import Root
    # from magicpandas.commandline.commandline import CommandLine


class Args:
    cache: dict[type, dict[str, list[Arg]]] = WeakKeyDictionary()
    # root: Root
    Root: Type[Root]

    def __get__(self, instance: Root, owner: type[Root]) -> dict[str, list[Arg]]:

        if owner not in self.cache:
            result: dict[str, list[Arg]] = collections.defaultdict(list)
            queue = collections.deque((owner,))
            i = 1
            # todo: allow for some args to keep their name if unable while contemporaries can
            while queue:
                obj: Frame | Type[Frame] = queue.popleft()
                if isinstance(obj, type):
                    cls = obj
                else:
                    cls = type(obj)
                for cls in cls.mro():
                    for key, value in cls.__dict__.items():

                        if isinstance(value, Frame):
                            value: Frame = getattr(obj, key)
                            assert value.mfirst is not None
                            queue.append(value)

                        if isinstance(value, Arg):
                            value = getattr(obj, key)
                            dest = value[i]
                            result[dest].append(value)

            self.cache[owner] = result
        return self.cache[owner]

    def dispatch(self):
        wrapped = [
            commandline.after
            for commandline in self.Root.magic.tree[CommandLine]
        ]
        argh.dispatch_commands(wrapped)


