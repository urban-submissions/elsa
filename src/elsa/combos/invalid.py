from __future__ import annotations

import tempfile
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa.resource import Resource

if False:
    from elsa.combos import Combos
    from elsa.combos.combos import Combos


class Invalid(Resource):
    __outer__: Combos

    def __from_inner__(self) -> Self:
        """Called when accessing Combos.invalid to instantiate Invalid"""
        combos = self.__outer__
        invalid = self.elsa.invalid
        _ = combos.normx, combos.normy
        order = 'label check ifile normx normy docstring isyns'.split()
        result = (
            combos
            ['ifile normx normy isyns'.split()]
            .merge(invalid, left_on='isyns', right_index=True)
            .sort_values('check isyns ifile normx normy'.split())
            .loc[:, order]
        )
        return result

    def write(self, path=None):

        if path is None:
            path = Path(tempfile.gettempdir(), 'disjoint.csv').resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_csv(path)
        print(str(path))
        return str(path)

    @magic.column
    def docstring(self) -> magic[str]:
        ...

    @magic.column
    def check(self) -> magic[str]:
        ...

    @magic.column
    def ifile(self) -> magic[int]:
        ...

    @magic.column
    def ibox(self) -> magic[int]:
        ...
