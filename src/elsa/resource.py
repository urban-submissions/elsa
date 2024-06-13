from __future__ import annotations

import warnings
from functools import *
from pandas import Series
from pathlib import Path
from typing import *
from typing import Self

import magicpandas as magic

if False:
    from .root import Elsa
    from .images import Images
    from .predictions import Predictions
    from .truth import Truth
    from .labels import Labels
    from .files import Files
    from .synonyms import Synonyms
    from elsa.annotation.prompts import Prompts
    from elsa.annotation.rephrase import Rephrase


class Resource(magic.Frame):
    """
    A base class with convenience attributes to be used for other
    frames in the library. These attributes allow for easy access to
    various objects regardless of location in the hierarchy.
    """
    __owner__: Elsa
    __outer__: Elsa
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]

    @magic.cached.cmdline.property
    def passed(self) -> Optional[Path, str]:
        """The path passed that will be used to construct the object."""
        return None

    file: Series

    @magic.column.from_options(dtype='category')
    def file(self) -> Series:
        """The file names of all unique images in all resources."""
        result = (
            self.elsa.images.file
            .loc[self.ifile]
            .values
        )
        return result

    @magic.column
    def path(self) -> Series[str]:
        result = (
            self.__outer__.files.path
            .loc[self.ifile]
            .values
        )
        return result

    @cached_property
    def elsa(self) -> Optional[Elsa]:
        from elsa.root import Elsa
        if isinstance(self, Elsa):
            return self
        outer = self.__outer__
        while True:
            if isinstance(outer, Elsa):
                return outer
            if isinstance(outer, Resource):
                return outer.elsa
            if outer is None:
                return None
            outer = outer.__outer__

    @magic.index
    def ifile(self) -> magic[str]:
        result = (
            self.elsa.images.file2ifile
            .loc[self.file]
            .values
        )
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        raise NotImplementedError

    @cached_property
    def images(self) -> Images:
        return self.elsa.images

    @cached_property
    def predictions(self) -> Predictions:
        return self.elsa.predictions

    @cached_property
    def truth(self) -> Truth:
        return self.elsa.truth

    @cached_property
    def labels(self) -> Labels:
        return self.elsa.labels

    @cached_property
    def files(self) -> Files:
        return self.elsa.files

    @cached_property
    def synonyms(self) -> Synonyms:
        return self.elsa.synonyms

    @cached_property
    def prompts(self) -> Prompts:
        return self.elsa.prompts

    @cached_property
    def rephrase(self) -> Rephrase:
        return self.elsa.truth.unique.rephrase

    @magic.column
    def source(self):
        return ''

    def __repr__(self):
        if (
                self.index.name == 'file'
                and 'ifile' in self.columns
        ):
            try:
                self = (
                    self
                    .reset_index('file', drop=True)
                    .set_index('ifile')
                )
            except:
                warnings.warn(
                    'Could not set index to "ifile".'
                )
        return super(Resource, self).__repr__()

    @magic.column
    def nfile(self) -> magic[int]:
        """An integer [0, N] for each unique ifile"""
        result = (
            self.elsa.images.nfile
            .loc[self.ifile]
            .values
        )
        return result
