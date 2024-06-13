from __future__ import annotations

import functools

import numpy as np
import tempfile
from pandas import DataFrame
from pandas import Series
from pathlib import Path
from typing import *
from typing import Self

import magicpandas as magic
from elsa.resource import Resource

if False:
    from .unique import Rephrase



class Prompts(Resource):
    """For each rephrase, aggregate the labels into a sentence"""
    unique: Self
    __outer__: Rephrase

    def __from_inner__(self) -> Self:
        index = self.__outer__.index.unique()
        result = self(index=index)
        _ = result.global_ilast, result.ifirst, result.ilast
        return result

    @magic.index
    def isyns(self) -> magic[int]:
        """Identifier for each unique combination of isyn"""

    @magic.index
    def irephrase(self) -> magic[int]:
        """Identifier for each rephrase for each unique combination of isyn"""

    @magic.column
    def natural(self) -> Series[str]:
        """
        The natural language
        """
        rephrase = self.__outer__
        _ = rephrase.iorder, rephrase.natural
        rephrase.natural.isna().any()

        result = (
            rephrase
            .reset_index()
            .groupby('isyns irephrase iorder'.split(), sort=False, observed=True)
            .natural
            .apply(' and '.join)
            .groupby('isyns irephrase'.split(), sort=False, observed=True)
            .apply(' '.join)
        )

        # append alone where only one subject

        return result

    @magic.column
    def cardinal(self) -> magic[str] | str:
        """Choose an arbitrary prompt to represent synonymous prompts"""
        result = (
            self.natural
            .groupby(self.isyns.values)
            .first()
            .loc[self.isyns]
            .values
        )
        return result

    @magic.column
    def cardinal(self) -> magic[str] | str:
        """Choose an arbitrary prompt to represent synonymous prompts"""
        _ = self.natural, self.isyns
        result = (
            self
            .sort_values('natural')
            .groupby(level='isyns', sort=False)
            .natural
            .first()
            .loc[self.isyns]
            .values
        )
        return result

    @magic.column
    def is_in_coco(self):
        """All the labels included in the prompt are in the COCO labels"""
        result = (
            self.__outer__.is_in_coco
            .groupby(level='isyns', sort=False)
            .all()
            .loc[self.isyns]
            .values
        )
        return result

    @magic.column
    def is_like_coco(self):
        """All of the labels included in the prompt are synonymous with COCO labels"""
        result = (
            self.__outer__.is_like_coco
            .groupby(level='isyns', sort=False)
            .all()
            .loc[self.isyns]
            .values
        )
        return result

    @magic.column
    def ifirst(self):
        return 0

    @magic.column
    def ilast(self):
        result = (
                self.natural.str.len()
                # exclude the period
                - 1
        )
        return result

    @magic.column
    def global_ilast(self):
        result = self.irephrase * 256
        return result

    @magic.column
    def combo(self):
        rephrase = self.__outer__
        _ = rephrase.isyn, rephrase.label
        result = (
            rephrase
            .sort_values('isyn')
            .groupby(level='irephrase', sort=False)
            .label
            .apply(' '.join)
            .loc[self.irephrase]
            .values
        )
        return result

    @magic.column
    def is_in_truth(self) -> magic[bool]:
        """Which prompts are located in the ground truth annotations"""
        result = self.combo.isin(self.truth.combo).values
        return result

    def implicated(
            self,
            by: Union[DataFrame, Series, str, list, tuple],
            synonyms: bool = True,
    ) -> Series[bool]:
        """
        Determine which prompts are implicated by the frame (e.g. files)
        prompts.implicated_by(files)
        prompts.implicated_by([file, path/to/file.png]
        """

        if synonyms:
            if isinstance(by, (str, tuple)):
                by = [by]
            if isinstance(by, list):
                # files passed
                truth = self.truth
                loc = truth.file.isin(by)
                loc |= truth.path.isin(by)
                loc |= truth.combo.isin(by)
                loc |= truth.natural.isin(by)
                loc |= truth.path.isin(by)
                loc |= truth.file.isin(by)
                isyns = truth.isyns.loc[loc].values
                loc = self.isyns.isin(isyns)
            elif hasattr(by, 'isyns'):
                loc = self.isyns.isin(by.isyns)
            elif hasattr(by, 'file'):
                file = by.file
                truth = self.truth
                loc = truth.file.isin(file.values)
                isyns = truth.isyns.loc[loc].values
                loc = self.isyns.isin(isyns)
            elif isinstance(by, Series):
                loc = self.isyns.isin(by)
            else:
                raise NotImplementedError
        else:
            if hasattr(by, 'combo'):
                loc = self.combo.isin(by.combo)
            elif hasattr(by, 'file'):
                file = by.file
                truth = self.truth
                loc = truth.file.isin(file.values)
                loc = (
                    self.combo
                    .isin(truth.combo.loc[loc])
                )
            elif isinstance(by, Series):
                loc = self.combo.isin(by)
            else:
                raise NotImplementedError
        return loc

    @magic.column
    def file(self):
        raise AttributeError

    @magic.column
    def level(self) -> magic[str]:
        combos = self.truth.combos
        _ = combos.level
        loc = ~combos.isyns.duplicated()
        level = (
            combos
            .loc[loc]
            .set_index('isyns')
            .loc[self.isyns, 'level']
            .values
        )
        return level

    def includes(
            self,
            label: str | int = None,
            meta: str | int = None,
    ) -> Series[bool]:
        """Determine if a combo contains a label"""
        rephrase = self.__outer__
        if label and meta:
            raise ValueError('label and meta cannot both be provided')
        if label is not None:
            if isinstance(label, str):
                isyn = self.synonyms.label2isyn[label]
            elif isinstance(label, int):
                isyn = label
            else:
                raise TypeError(f'label must be str or int, not {type(label)}')
            loc = rephrase.isyn == isyn
        elif meta is not None:
            loc = rephrase.meta == meta
        else:
            raise ValueError('label or meta must be provided')

        result = (
            Series(loc)
            .groupby(rephrase.isyns, sort=False)
            .any()
            .loc[self.isyns]
        )
        return result

    def excludes(
            self,
            label: str | int = None,
            meta: str | int = None,
    ) -> Series[bool]:
        """Determine if a combo excludes a label"""
        return ~self.includes(label, meta)

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Pass a mask that is aligned with the annotations;
        group that mask by the isyns and count the number of unique labels
        """
        if loc is None:
            loc = slice(None)
        rephrase = self.__outer__
        result = Series(0, index=self.isyns)
        isyns = rephrase.isyns[loc]
        update = (
            rephrase.isyn
            .loc[loc]
            .groupby(isyns)
            .nunique()
        )
        result.update(update)
        # result = result.values
        return result

    def write(self, path: str = None):
        """Write all unique prompts to a file"""
        if path is None:
            path = Path(tempfile.gettempdir(), 'prompts.txt').resolve()
        prompts: list[str] = self.natural.unique().tolist()
        outpath = Path(__file__, '..', path).resolve()
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, 'w+') as f:
            for prompt in prompts:
                f.write(f"{prompt}\n")
        print(outpath)

    @magic.series
    def nprompt(self):
        natural = self.natural.drop_duplicates()
        result = (
            Series(np.arange(len(natural)), index=natural)
            .loc[self.natural]
            .set_axis(self.index)
        )
        return result

    @magic.series
    def natural2level(self) -> Series:
        _ = self.level
        result = (
            self.level
            .set_axis(self.natural)
        )
        return result

    @magic.column
    def condition(self) -> magic[str]:
        """Represents which condition out of {person, pair, people} the prompt has."""
        person = self.includes('person')
        pair = self.includes('pair')
        people = self.includes('people')
        none = ~person & ~pair & ~people
        assert (
            pair
            ^ person
            ^ people
            | none
        ) .all()
        result = np.full_like(person, '', dtype=object)
        result = np.where(person, 'person', result)
        result = np.where(pair, 'pair', result)
        result = np.where(people, 'people', result)
        return result


