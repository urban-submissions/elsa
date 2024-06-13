from __future__ import annotations

import json
import numpy as np
import pandas as pd
from functools import cached_property
from numpy import ndarray
from pandas import Series
from pathlib import Path
from typing import *

import magicpandas as magic
import elsa.util as util
from elsa.resource import Resource

bing = Path(__file__, *'.. static bing label_ids.txt'.split()).resolve()
google = Path(__file__, *'.. static google label_ids.txt'.split()).resolve()
unified = Path(__file__, *'.. .. .. gt_data triple_inspected_May23rd merged label_id_dict_after_distr_thresholding.csv'.split()).resolve()


class Labels(Resource):

    @magic.column
    def isyn(self) -> Series[int]:
        """
        The index of the set of synonyms that the label belongs to;
        this resolves ambiguities.
        """
        result = (
            self.synonyms.isyn
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def ibox(self) -> Series[int]:
        """Unique index for each combo in each file"""

    @magic.cached.property
    def year(self) -> int:
        raise AttributeError

    @magic.cached.property
    def version(self) -> str:
        raise AttributeError

    @magic.cached.property
    def contributor(self) -> str:
        raise AttributeError

    @magic.index
    def ilabel(self):
        ...

    @magic.column
    def label(self) -> Series[str]:
        """The name given to each label"""

    @magic.column
    def multi(self) -> Series[str]:
        ...

    @magic.column
    def color(self):
        """Map a color to each label"""
        colors = util.colors
        assert len(self) <= len(colors)
        return colors[:len(self)]

    # todo: label, ilabel
    @classmethod
    def from_json(cls, path):
        with open(path) as file:
            data = json.load(file)
        cats: list[dict[str, str | int]] = data['categories']
        info = data['info']
        count = len(cats)
        ilabel = np.fromiter((
            cat['id']
            for cat in cats
        ), int, count=count)
        label = np.fromiter((
            cat['name'].casefold()
            for cat in cats
        ), object, count=count)

        index = pd.Index(ilabel, name='ilabel')
        result = cls(dict(
            label=label,
        ), index=index)
        result.year = info['year']
        result.version = info['version']
        result.contributor = info['contributor']
        return result

    @classmethod
    def from_txt(cls, path) -> Self:
        with open(path) as file:
            lines = file.readlines()
        count = len(lines)
        ilabel = np.arange(count)
        label = np.fromiter((
            line.strip()
            for line in lines
        ), object, count=count)
        index = pd.Index(ilabel, name='ilabel')
        result = cls(dict(
            label=label,
        ), index=index)
        return result

    @classmethod
    def from_csv(cls, path) -> Self:
        columns = dict(
            id='ilabel',
        )
        result = (
            pd.read_csv(path)
            .rename(columns=columns)
            .pipe(cls)
        )
        return result

    @classmethod
    def from_pickle(cls, path) -> Self:
        result = cls.from_pickle(path)
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        if isinstance(path, (Path, str)):
            path = Path(path)
            match path.suffix:
                case '.json':
                    result = cls.from_json(path)
                case '.csv':
                    result = cls.from_csv(path)
                case '.txt':
                    result = cls.from_txt(path)
                case _:
                    raise ValueError(f'Unsupported file type: {path.suffix}')
        elif path is None:
            result = cls()
        else:
            msg = f'Labels expected a Path or str, got {type(path)}'
            raise TypeError(msg)
        result.passed = path
        return result

    def __from_inner__(self) -> Self:
        with self.configure:
            passed = self.passed
        result = self.from_inferred(passed)
        result.label = result.label.str.casefold()
        return result

    @magic.column
    def meta(self) -> Series[str]:
        result = (
            Series(self.label2meta)
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def imeta(self) -> Series[int]:
        result = (
            Series(self.meta2imeta)
            .loc[self.meta]
            .values
        )
        return result

    def decompose(self, labels: list[str]) -> list[list[list[str]]]:
        wrong2ambuigities = self.wrong2ambiguities
        result: list[list[list[str]]] = []
        LABEL: str
        for LABEL in labels:
            possibilities = [[]]
            label = str(LABEL)
            while label:
                for wrong, ambiguities in wrong2ambuigities.items():
                    if label.startswith(wrong):
                        label = (
                            label
                            .replace(wrong, '')
                            .strip()
                        )
                        possibilities = [
                            possibility + [amguity]
                            for possibility in possibilities.copy()
                            for amguity in ambiguities
                        ]
                        break
                else:
                    raise ValueError(f'Could not decompose {LABEL=}')
            result.append(possibilities)

        return result

    def get_ilabel(self, label: list[str] | Series[str]) -> ndarray[int]:
        result = (
            self
            .reset_index()
            .set_index('label')
            .ilabel
            .loc[label]
            .values
        )
        return result

    @cached_property
    def label2ilabel(self) -> Series[int]:
        result = Series(self.ilabel.values, index=self.label, name='ilabel')
        return result

    @magic.column
    def is_in_prompts(self):
        result = self.label.isin(self.synonyms.prompts.syn)
        return result

    @magic.column
    def is_synonymous_in_prompts(self):
        result = self.isyn.isin(self.synonyms.prompts.isyn)
        return result

    @magic.column.from_options(dtype='category')
    def meta(self) -> Series[str]:
        """The metaclass, or type of label, for each box"""
        result = self.synonyms.meta.loc[self.label].values
        return result
