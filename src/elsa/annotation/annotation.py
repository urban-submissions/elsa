from __future__ import annotations

import functools
import numpy as np
from pandas import Series
from types import *
from typing import *
from typing import Self

import magicpandas as magic
from elsa.annotation.unique import Unique
from elsa.boxes import Boxes
from elsa.combos.combos import Combos


class Annotation(Boxes):
    loc: MappingProxyType[Any, Self] | Self
    iloc: MappingProxyType[Any, Self] | Self
    unique = Unique()

    @Combos
    def combos(self) -> Combos:
        """
        A DataFrame encapsulating the annotations aggregated by box.
        Three annotations representing 'person', 'walking', and 'on phone'
        will be aggregated into a single combo entry, with the label
        being 'person walking on phone'.
        """

    @magic.column
    def isyn(self) -> Series[int]:
        """
        Synonym ID of the label; synonymous labels will have the same
        isyn value. For example, 'person' and 'individual' have the same
        isyn value.
        """
        result = (
            self.elsa.synonyms.isyn
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def isyns(self) -> Series[tuple[int]]:
        """
        An ordered tuple of the synonym IDs representing a given combo.
        For example, "person walking" would have isyns=(0, 1), and
        "pedestrian strolling" would also have isyns=(0, 1), as they are
        synonymous.
        """
        result = (
            self.combos.isyns
            .subloc[self.ibox]
            .values
        )
        return result

    @magic.index
    def iann(self) -> magic[int]:
        """Each annotation has a unique identifier integer: iann"""
        return np.arange(len(self))

    @magic.column
    def ilabel(self) -> Series[int]:
        """Label ID for each annotation, assigned by the dataset"""
        labels = self.labels
        if not labels.passed:
            raise AttributeError
        result = (
            Series(labels.ilabel, index=labels.label)
            .loc[self.label]
            .values
        )
        return result

    @magic.column.from_options(dtype='category')
    def label(self) -> magic[str]:
        """String label for each annotation assigned by the dataset"""
        labels = self.labels
        if not labels.passed:
            raise AttributeError
        _ = self.ilabel
        result = labels.label.loc[self.ilabel].values

        return result

    @magic.column.from_options(dtype='category')
    def meta(self) -> Series[str]:
        """Metaclass, or type of label, for each box"""
        result = self.synonyms.meta.loc[self.label].values
        return result

    @magic.column
    def ibox(self) -> Series[int]:
        """
        Unique identifier for each combo box that the annotatin belongs
        to in aggregate
        """
        _ = self.combos
        return self.ibox

    @magic.column.from_options(dtype='category')
    def combo(self) -> magic[str]:
        """
        String of combined labels for the combo box that the annotation
        belongs to in aggregate:

        label       combo
        person      person walking on phone
        walking     person walking on phone
        on phone    person walking on phone
        """
        _ = self.combos
        return self.combo

    def includes(
            self,
            label: str = None,
            meta: str = None,
    ) -> Series[bool]:
        """
        Return a boolean mask for annotations that belong to a combo
        that include the label or metalabel provided.
        """
        if label and meta:
            raise ValueError('label and meta cannot both be provided')
        if label is not None:
            if isinstance(label, str):
                isyn = self.synonyms.label2isyn[label]
            elif isinstance(label, int):
                isyn = label
            else:
                raise TypeError(f'label must be str or int, not {type(label)}')
            loc = self.isyn == isyn
        elif meta is not None:
            loc = self.meta == meta
        else:
            raise ValueError('label or meta must be provided')

        result = (
            Series(loc)
            .groupby(self.ibox, sort=False)
            .any()
            .loc[self.ibox]
            .set_axis(self.index)
        )
        return result

    def excludes(
            self,
            label: str = None,
            meta: str = None,
    ) -> Series[bool]:
        """
        Return a boolean mask for annotations that belong to a combo
        that do not include the label or metalabel provided.
        """
        result = ~self.includes(label, meta)
        return result

    def synonymous(self, label: str) -> Series[bool]:
        """True if the label is synonymous with the passed label"""
        isyn = self.synonyms.isyn.loc[label]
        loc = self.isyn.values == isyn
        return loc

    @magic.column
    def iorder(self) -> magic[int]:
        """
        Value assigned for ordering the labels in the process of
        generating natural prompts from the combinations.
        """
        result = (
            self.elsa.synonyms.iorder
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def natural(self) -> magic[str]:
        """
        Natural language representation of the label:
        'person' -> 'a person'
        'sports' -> 'doing sports'
        """
        result = (
            self.elsa.synonyms.natural
            .loc[self.label]
            .values
        )
        return result

    @magic.column
    def is_invalid(self):
        """True where annotation is part of an invalid combo"""
        return self.combos.is_invalid.loc[self.ibox].values

    def indexed(self) -> Self:
        result = self
        _ = result.iann
        result = result.set_index('iann')
        return result

    @functools.cached_property
    def yolo(self) -> Self:
        """Return the annotations in YOLO format"""
        _ = self.normx, self.normy, self.normheight, self.normwidth, self.path
        columns = ['normx', 'normy', 'normheight', 'normwidth', 'path']
        return self[columns]
