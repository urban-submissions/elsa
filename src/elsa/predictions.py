from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

import numpy as np
import os
import pandas as pd
import warnings
from ast import literal_eval
from numpy import ndarray
from pandas import Series
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa.annotation import Annotation

if False:
    from .root import Elsa

class Predictions(Annotation):
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]
    __owner__: Elsa
    __outer__: Elsa

    # def __align__(self, owner: Elsa = None) -> Self:
    #     from elsa.root import Elsa
    #     if not isinstance(owner, Elsa):
    #         return self
    #     loc = self.ifile.isin(owner.ifile)
    #     result = self.loc[loc]
    #     return result


    @magic.column
    def confidence(self) -> Series[float]:
        """prediction confidence"""

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> Self:
        frame = pd.read_csv(path)
        if pd.Series(
            'ifile w s e n ilabel label confidence'.split()
        ).isin(frame.columns).all():
            result = cls(frame)
        elif pd.Series(
                'ifile x y width height ilabel num_labels is_valid'.split()
        ).isin(frame.columns).all():
            result = cls.from_new(frame)
        elif 'data_source' in frame.columns:
            result = cls.from_unified(frame)
        else:
            result = cls.from_old(frame)
        result.passed = path
        return result

    @classmethod
    def from_old(cls, frame: pd.DataFrame, ) -> Self:
        """Old predictions CSV format"""
        file = (
            frame.img_path.str
            .rsplit(os.sep, expand=True, n=1)
            .iloc[:, 1]
            .values
        )
        label = frame.columns[2:]
        ncols = len(label)
        nrows = len(frame)

        strings = frame.iloc[:, 2:].values.ravel()
        count = ncols * nrows

        label_records: list[list] = list(map(literal_eval, strings))
        size: ndarray = np.fromiter(map(len, label_records), int, count)
        repeat = (
            size
            .reshape(nrows, ncols)
            .sum(axis=1)
        )
        file = file.repeat(repeat)
        columns = 'label confidence w s e n'.split(' ')
        result = cls.from_records((
            (label, confidence, w, s, e, n)
            for records in label_records
            for label, confidence, (w, s, e, n) in records
        ), columns=columns)
        result['file'] = file
        return result

    @classmethod
    def from_new(cls, frame: pd.DataFrame) -> Self:
        """New predictions CSV format"""
        list_ilabels = list(map(literal_eval, frame.ilabel))
        repeat = np.fromiter(map(len, list_ilabels), int, count=len(list_ilabels))
        ilabel = np.concatenate(list_ilabels)
        iloc = np.arange(len(frame)).repeat(repeat)
        columns = dict(
            x='normx',
            y='normy',
            width='normwidth',
            height='normheight',
        )
        result = (
            frame
            .assign(num_labels=repeat)
            .iloc[iloc]
            .assign(ilabel=ilabel)
            .rename(columns=columns)
            .pipe(cls)
        )
        return result

    @classmethod
    def from_unified(cls, frame: pd.DataFrame) -> Self:
        list_ilabels = list(map(literal_eval, frame.ilabel))
        repeat = np.fromiter(map(len, list_ilabels), int, count=len(list_ilabels))
        ilabel = np.concatenate(list_ilabels)
        iloc = np.arange(len(frame)).repeat(repeat)
        columns = dict(
            x='normx',
            y='normy',
            width='normwidth',
            height='normheight',
            unique_ifile='ifile',
        )
        result = (
            frame
            .assign(num_labels=repeat)
            .iloc[iloc]
            .assign(ilabel=ilabel)
            .rename(columns=columns)
            .pipe(cls)
        )
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        path = Path(path)
        if isinstance(path, pd.DataFrame):
            result = cls(path)
            result.passed = None
            return result

        if path.is_dir():
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(cls.from_inferred, file) for file in path.iterdir() if file.is_file()]
                results = [future.result().data for future in futures]
            concatenated_data = pd.concat(results, ignore_index=True)
            result = cls(concatenated_data)
            result.passed = path
            return result

        match path.suffix:
            case '.csv':
                result = cls.from_csv(path)
            case _:
                try:
                    func = getattr(cls, f'from_{path.suffix[1:]}')
                except AttributeError:
                    ...
                else:
                    result = func(path)
                    result = cls(result)
                    result.passed = path
                    return result
                try:
                    func = getattr(pd, f'read_{path.suffix[1:]}')
                except AttributeError:
                    raise ValueError(f'Unsupported file type: {path.suffix}')
                else:
                    frame = func(path)
                    result = cls(frame)
                    result.passed = path
                    return result

        return result

    @magic.column
    def natural(self) -> Series[str]:
        """
        The natural multilabel used to generate the prediction, if
        applicable
        """
        # noinspection PyTypeChecker
        return ''


    def __from_inner__(self) -> Self:
        with self.configure:
            passed = self.passed
        result = (
            self
            .from_inferred(passed)
            .pipe(self)
            .indexed()
        )

        _ = result.fw, result.fs, result.fn, result.fe
        bounds = 'fw fs fn fe'.split()
        loc = result[bounds].max(axis=1) - result[bounds].min(axis=1) <= 1.
        assert loc.all(), 'file-specific bounds are not consistent'

        # # drop model_hint
        # isyn = result.synonyms.label2isyn['model_hint']
        # loc = result.isyn != isyn
        # result = result.loc[loc]

        labels = result.labels
        loc = result.ilabel.isin(labels.ilabel)
        if not loc.all():
            unique = result.ilabel[~loc].unique()
            warnings.warn(
                f'{len(unique)} ilabels {unique} from '
                f'{result.passed} do not occur in labels '
                f'metadata {labels.passed}; dropping.'
            )
            result = result.loc[loc]

        return result


bing = Path(__file__, *'.. static bing predictions.csv'.split()).resolve()
google = Path(__file__, *'.. static google predictions.csv'.split()).resolve()
unified = None
if __name__ == '__main__':
    pred = Predictions.from_inferred(bing)
