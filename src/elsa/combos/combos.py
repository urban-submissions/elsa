from __future__ import annotations

import PIL
import itertools
import networkx
import numpy as np
import numpy.core.defchararray
import os
import pandas as pd
from PIL import Image, ImageDraw
from functools import cached_property
from functools import reduce
from numpy import ndarray
from pandas import Series
from pathlib import Path
from typing import *
from typing import Self

import magicpandas as magic
import elsa.util as util
from elsa.boxes import Boxes
from elsa.combos.invalid import Invalid

if False:
    from elsa.annotation import Annotation


class Combos(Boxes):
    __owner__: Annotation
    __outer__: Annotation
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]

    @magic.index
    def ibox(self) -> Series[int]:
        """Unique index for each combo entry"""
        result = np.arange(len(self))
        # noinspection PyTypeChecker
        return result

    @magic.column
    def label(self) -> Series[str]:
        """Multilabel from the joined labels"""

    @Invalid
    def invalid(self):
        """
        A DataFrame that broadcasts Elsa.invalid to each combo in the
        annotations, representing which annotations are invalid and why.
        """

    @magic.column
    def is_invalid(self) -> Series[bool]:
        """
        Whether the boxes are invalid.
        invalid boxes are considered false positives.
        """
        unique = self.__outer__.unique.consumed
        invalid = unique.invalid.all_checks.any(axis=1)
        isyns = invalid[invalid].index
        isyns = unique.isyns.loc[isyns]
        loc = self.isyns.isin(isyns)
        return loc

    @magic.column
    def nlabels(self) -> Series[int]:
        """How many labels are in the combo"""
        result = (
            self.__outer__
            .groupby('ibox', sort=False)
            .size()
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def nunique_labels(self):
        """How many unique labels are in the combo"""
        outer = self.__outer__
        result = (
            outer.isyn
            .groupby(outer.ibox, sort=False)
            .nunique()
            .loc[self.ibox]
            .values
        )
        return result

    @magic.cached.cmdline.property
    def threshold(self) -> float:
        """iou threshold"""
        return .9

    def __from_inner__(self) -> Self:
        """
        Called when accessing Annotation.combos to instantiate Combos
        """
        inner: Combos = self
        g = networkx.Graph()
        BOXES = self.__outer__
        boxes = self.__outer__
        arrays = boxes.fw, boxes.fs, boxes.fe, boxes.fn
        _ = boxes.geometry
        names = 'fw fs fe fn'.split()
        bounds = pd.MultiIndex.from_arrays(arrays, names=names)

        # save some time by only considering unique boxes
        loc = ~bounds.duplicated()
        unique = bounds[loc]
        iann: Series = boxes.iann
        ifirst = (
            Series(iann[loc].values, index=unique)
            .loc[bounds]
        )
        edges = zip(ifirst, iann)
        g.add_edges_from(edges)

        cc = list(networkx.connected_components(g))
        repeat = np.fromiter(map(len, cc), int, len(cc))
        ibox = (
            np.arange(len(repeat))
            .repeat(repeat)
        )
        it = itertools.chain.from_iterable(cc)
        data: ndarray = np.fromiter(it, int, repeat.sum())
        iann = Series(data, name='iann')

        # update ibox for boxes
        annotations = self.__outer__
        annotations.ibox = -1
        # annotations.subloc[iann, 'ibox'] = ibox
        annotations.loc[iann, 'ibox'] = ibox
        assert (annotations.ibox != -1).all()
        _ = annotations.isyn, annotations.label
        label = (
            annotations
            .drop_duplicates('ibox isyn'.split())
            .sort_values('isyn')
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        annotations.combo = None
        annotations.subloc[iann, 'combo'] = label.loc[ibox].values

        _ = annotations.file
        groupby = (
            annotations
            .reset_index()
            .groupby('ibox', sort=False)
        )
        fw = groupby.fw.min()
        fs = groupby.fs.min()
        fe = groupby.fe.max()
        fn = groupby.fn.max()
        assert (fw <= fe).all()
        assert (fs <= fn).all()
        ifile = groupby.ifile.first()
        file = groupby.file.first()
        result = inner(dict(
            label=label,
            fw=fw,
            fs=fs,
            fe=fe,
            fn=fn,
            file=file,
            ifile=ifile,
        ))
        result = (
            result
            .reset_index()
            .set_index('ibox')
        )
        return result

    @magic.column
    def rcombo(self) -> Series[int]:
        """
        The relative index of each combo for that
        file; the lowest index is 0 for each file e.g.
        0 1 2 3 0 1 0 1 2 0 0 rcombo
        a a a a b b c c c d d file
        """
        arrays = self.ifile, self.ibox
        names = 'ifile ibox'.split()
        needles = pd.MultiIndex.from_arrays(arrays, names=names)
        haystack = needles.unique().sort_values()
        ifile = haystack.get_level_values('ifile')
        con = util.constituents(ifile)
        ifirst = con.ifirst.repeat(con.repeat)
        rcombo = np.arange(len(haystack)) - ifirst
        result = (
            Series(rcombo, index=haystack)
            .loc[needles]
            .values
        )
        return result

    @magic.column
    def color(self) -> Series[str]:
        """The color for each box"""
        rcombo = self.rcombo
        assert rcombo.max() < len(util.colors)
        result = (
            Series(util.colors)
            .iloc[rcombo]
            .values
        )
        return result

    def print(
            self,
            ifile: str | int = None,
            columns: str = tuple('color label'.split()),
    ):
        # todo: also add level, ilevel, ilabel
        for col in columns:
            getattr(self, col)
        columns = list(columns)
        with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
        ):
            sub = self.at_ifile(ifile)
            print(sub[columns])

    def view(
            self,
            ifile: str | int = None,
            columns: str = tuple('color label'.split()),
    ) -> PIL.Image:
        """
        Given a string filename or integer ifile,
        visualize the
        """
        _ = self.label, self.color, self.w, self.s, self.e, self.n
        if isinstance(ifile, (str, Path)):
            ifile = (
                str(ifile)
                .rsplit(os.sep, 1)[-1]
                .split('.', 1)[0]
            )

        sub = self.at_ifile(ifile)
        if not len(sub):
            raise ValueError(f'No boxes for {ifile=}')
        ifile = sub.ifile.values[0]
        width = sub.images.width.at[ifile]
        height = sub.images.height.at[ifile]
        size = width, height
        if sub.files.passed:
            path = sub.files.path.at[ifile]
            result = Image.open(path)
        else:
            result = Image.new('RGB', size, 'black')
        draw = ImageDraw.Draw(result)

        w = sub.w.astype(int)
        s = sub.s.astype(int)
        e = sub.e.astype(int)
        n = sub.n.astype(int)
        colors = sub.color
        its = w, s, e, n, colors
        for w, s, e, n, color in zip(*its):
            draw.rectangle([w, s, e, n], outline=color)
        sub.print(ifile, columns=columns)

        return result

    def views(self, loc=None) -> Iterator[PIL.Image]:
        """
        Given an optional mask, iteratively visualize each image
        that matches the mask.
        """
        if loc is None:
            loc = slice(None)
        ifiles = self.ifile[loc].unique()
        for ifile in ifiles:
            yield self.view(ifile)

    def at_ifile(self, ifile: str = None) -> Self:
        """ Return a view of the combos at a given file """
        if ifile is None:
            ifile = self.ifile.values[0]
        loc = self.ifile == ifile
        if not loc.any():
            loc = self.file == ifile
            if not loc.any():
                raise ValueError(f'No boxes for {ifile=}')
        self = self.loc[loc]
        return self

    def at_file(self, file: str) -> Self:
        """ Return a view of the combos at a given file """
        if file is None:
            file = self.file.values[0]
        loc = self.file == file
        if not loc.any():
            raise ValueError(f'No boxes for {file=}')
        self = self.loc[loc]
        return self

    def includes(
            self,
            label: str | int = None,
            meta: str | int = None,
    ) -> Series[bool]:
        """True where a combo contains a label"""
        ann = self.__outer__
        if label and meta:
            raise ValueError('label and meta cannot both be provided')
        if label is not None:
            if isinstance(label, str):
                isyn = self.synonyms.label2isyn[label]
            elif isinstance(label, int):
                isyn = label
            else:
                raise TypeError(f'label must be str or int, not {type(label)}')
            loc = ann.isyn == isyn
        elif meta is not None:
            loc = ann.meta == meta
        else:
            raise ValueError('label or meta must be provided')

        result = (
            Series(loc)
            .groupby(ann.ibox, sort=False)
            .any()
            .loc[self.ibox]
        )
        return result

    def excludes(
            self,
            label: str | int = None,
            meta: str | int = None,
    ) -> Series[bool]:
        """True where a combo excludes a label"""
        return ~self.includes(label, meta)

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        """
        Given a mask that is aligned with the annotations,
        determine the number of unique labels in each combo
        """
        if loc is None:
            loc = slice(None)
        ann = self.__outer__
        result = Series(0, index=self.ibox)
        ibox = ann.ibox[loc]
        update = (
            ann.isyn
            .loc[loc]
            .groupby(ibox)
            .nunique()
        )
        result.update(update)
        result = result.set_axis(self.index)
        return result

    @magic.column
    def level(self) -> magic[str]:
        """The level of the combo, e.g. cs, csa, csao"""
        # todo: must we require cs if a? must we require csa if o?
        includes = self.includes
        loc = includes(meta='condition')
        condition = np.where(loc, 'c', '')
        loc = includes(meta='state')
        state = np.where(loc, 's', '')
        loc = includes(meta='activity')
        activity = np.where(loc, 'a', '')
        loc = includes(meta='others')
        others = np.where(loc, 'o', '')
        sequence = condition, state, activity, others
        result = reduce(np.core.defchararray.add, sequence)
        return result

    # @magic.column
    # def level(self) -> Series[str]:
    #     """The level of the combo, e.g. cs, csa, csao"""
    #     result = Series('', index=self.ibox)
    #     includes = self.includes
    #     loc = includes(meta='condition')
    #     result.loc[loc] = 'c'
    #     loc &= includes(meta='state')
    #     result.loc[loc] = 'cs'
    #     loc &= includes(meta='activity')
    #     result.loc[loc] = 'csa'
    #     loc &= includes(meta='others')
    #     result.loc[loc] = 'csao'
    #     result = result.values
    #     return result

    @magic.column
    def ilevel(self) -> Series[int]:
        """
        The integer level of the combo:
        c: 1,
        cs: 2,
        csa: 3,
        csao: 4
        """
        result = Series(0, index=self.ibox)
        includes = self.includes
        loc = includes(meta='condition')
        result.loc[loc] = 1
        loc &= includes(meta='state')
        result.loc[loc] = 2
        loc &= includes(meta='activity')
        result.loc[loc] = 3
        loc &= includes(meta='others')
        result.loc[loc] = 4
        result = result.values
        return result

    @magic.column
    def isyns(self) -> Series[tuple[int]]:
        """An ordered tuple of the isyns associated with the combo"""

        ann = self.__outer__
        _ = ann.ibox, ann.isyn
        result = (
            ann
            .reset_index()
            .sort_values('isyn')
            .groupby('ibox', sort=False)
            .isyn
            .apply(tuple)
            .loc[self.ibox]
            .values
        )
        for t in result:
            assert all(t[i] <= t[i + 1] for i in range(len(t) - 1))
        return result

    @magic.column
    def c(self) -> Series[str]:
        """
        The combo label with only the condition;
        None if no condition is present
        """
        result = Series(None, index=self.ibox, dtype=object)
        ann = self.__outer__
        loc = ann.meta.values == 'condition'
        loc &= (
            self.includes(meta='condition')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @magic.column
    def cs(self) -> Series[str]:
        """
        The combo label with only the condition and state;
        None if condition or state are not present
        """
        result = Series(None, index=self.ibox, dtype=object)
        ann = self.__outer__
        loc = ann.meta.values == 'condition'
        loc |= ann.meta.values == 'state'
        loc &= (
            self.includes(meta='condition')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(meta='state')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @magic.column
    def csa(self) -> Series[str]:
        """
        The combo label with only the condition, state, and activity;
        None if condition, state, or activity are not present
        """
        result = Series(None, index=self.ibox, dtype=object)
        ann = self.__outer__
        loc = ann.meta.values == 'condition'
        loc |= ann.meta.values == 'state'
        loc |= ann.meta.values == 'activity'
        loc &= (
            self.includes(meta='condition')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(meta='state')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(meta='activity')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @magic.column
    def csao(self) -> Series[str]:
        """
        The combo label with the condition, state, activity, and others;
        None if condition, state, activity, or others is not present
        """
        result = Series(None, index=self.ibox, dtype=object)
        ann = self.__outer__
        loc = ann.meta.values == 'condition'
        loc |= ann.meta.values == 'state'
        loc |= ann.meta.values == 'activity'
        loc |= ann.meta.values == 'others'
        loc &= (
            self.includes(meta='condition')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(meta='state')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(meta='activity')
            .loc[ann.ibox]
            .values
        )
        loc &= (
            self.includes(meta='others')
            .loc[ann.ibox]
            .values
        )
        update = (
            ann
            .loc[loc]
            .groupby('ibox', sort=False)
            .label
            .apply(' '.join)
        )
        result.update(update)
        result = result.values
        return result

    @cached_property
    def annotations(self) -> Annotation:
        return self.__outer__


    @magic.column
    def cardinal(self) -> magic[str]:
        ...

