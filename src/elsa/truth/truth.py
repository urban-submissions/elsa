from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import csv
import glob
import numpy as np
import os
import pandas as pd
import warnings
from ast import literal_eval
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa import util
from elsa.annotation import Annotation
from elsa.truth.combos import Combos

if False:
    import elsa.truth.dropped
    from elsa import Elsa
    import elsa.truth.upgrade

bing = {'BSV_': Path(__file__, *'.. .. static bing bing_gt.csv'.split()).resolve(), }
google = {'GSV_': Path(__file__, *'.. .. static google google_gt.csv'.split()).resolve(), }
original = bing | google
unified = Path(__file__, *'.. .. .. .. gt_data triple_inspected_May23rd merged label_per_box_sanity_checked_removed_unwanted_labels_unified_labels_after_distr_thresholding.csv'.split()).resolve()


def read_csv(
        file_path: str,
        fallback: list[str] = None,
):
    with open(file_path, 'r') as file:
        sample = file.read(1024)
        sniffer = csv.Sniffer()
        sep = sniffer.sniff(sample).delimiter

    with open(file_path, 'r') as file:
        names = file.readline().split(sep)
        if not all(
                name[0].isalpha() or name[0] == '_'
                for name in names
        ):
            names = fallback
            df = pd.read_csv(file_path, sep=sep, names=names)
            df['irow'] = np.arange(len(df))
        else:
            names = None
            df = pd.read_csv(file_path, sep=sep, names=names)
            df['irow'] = np.arange(len(df)) + 1
    return df


class Truth(Annotation):
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]
    dropped: elsa.truth.dropped.Dropped
    combos: Combos = Combos()
    __outer__: Elsa
    __owner__: Elsa

    def dropped(self) -> elsa.truth.dropped.Dropped:
        """DataFrame that describes data regarding which labels were dropped."""

    @classmethod
    def from_inferred(cls, paths) -> Self:
        if isinstance(paths, (Path, str)):
            paths = Path(paths)
            if paths.is_dir():
                with ThreadPoolExecutor() as threads:
                    it = (
                        file
                        for file in paths.iterdir()
                        if file.is_file()
                           and os.path.getsize(file)
                    )
                    futures = threads.map(cls.from_inferred, it)
                    results = list(futures)
                result = pd.concat(results, ignore_index=True)
                result.file = util.trim_path(result.file)
                result = cls(result)
                result.passed = paths
                return result
            if (
                    str(paths).endswith('.txt')
                    or str(paths).endswith('.csv')
            ):
                result = cls.from_csv(paths)
            else:
                paths = os.path.join(paths, '*.txt')
                options = glob.glob(paths)
                paths = os.path.join(paths, '*.csv')
                options += glob.glob(paths)
                paths = options
                result = cls.from_csvs(paths)
        elif isinstance(paths, dict):
            concat = []
            for source, path in paths.items():
                if (
                        str(path).endswith('.txt')
                        or str(path).endswith('.csv')
                ):
                    result = cls.from_csv(path)
                else:
                    path = os.path.join(path, '*.txt')
                    options = glob.glob(path)
                    path = os.path.join(path, '*.csv')
                    options += glob.glob(path)
                    path = options
                    result = cls.from_csvs(path)
                ifile = str(source) + result.ifile.astype(str)
                result = result.assign(ifile=ifile, source=source)
                concat.append(result)
            result = pd.concat(concat, ignore_index=True)
        elif isinstance(paths, Iterable):
            paths = list(paths)
            result = cls.from_csvs(paths)
        elif paths is None:
            result = cls()
            return result
        else:
            raise TypeError(f'Expected a Path, str, or Iterable, got {type(paths)}')
        # result = result.pipe(cls)
        if 'data_source' in result.columns:
            result = cls.from_new(result)
        # currently multithreaded assignments break magicpandas caches
        if 'file' in result:
            result = pd.DataFrame(result)
            result.file = util.trim_path(result.file)
        result = cls(result)
        result.passed = paths
        return result

    @classmethod
    def from_csv(cls, csv: str) -> Self:
        """Create a Truth object from a csv file."""
        fallback = 'ilabel normx normy normwidth normheight'.split()
        result = read_csv(csv, fallback=fallback)
        if (
                'file' not in result
                and 'ifile' not in result
        ):
            file = (
                str(csv)
                .rsplit(os.sep, maxsplit=1)
                [-1]
                .split('.')
                [0]
            )
            result = result.assign(file=file, path=csv)
            result.file = util.trim_path(result.file)

        result = cls(result)
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
            # unique_ifile='ifile'
        )
        result = (
            frame
            .assign(ifile=frame.unique_ifile)
            .assign(num_labels=repeat)
            .iloc[iloc]
            .assign(ilabel=ilabel)
            .rename(columns=columns)
            # .pipe(cls)
        )
        result['source'] = (
            result.data_source.str
            .casefold()
            .str[:1]
        )
        # result.file = util.trim_path(result.file)
        # result = cls(result)
        return result

    @classmethod
    def from_csvs(cls, csvs: list[str]) -> Self:
        """Create a Truth object from a list of csv files."""

        def read_csv(csv: str):
            file = (
                str(csv)
                .rsplit(os.sep, maxsplit=1)
                [-1]
                .replace('txt', 'png')
                .replace('csv', 'png')
            )
            names = 'ilabel normx normy normwidth normheight'.split()
            result = (
                pd.read_csv(csv, sep=' ', names=names, )
                .assign(file=file)
            )
            return result

        with ThreadPoolExecutor() as threads:
            frames = [
                frame
                for frame in threads.map(read_csv, csvs)
                if len(frame)
            ]
        result = (
            pd.concat(frames, ignore_index=True)
            # .pipe(cls)
        )
        result.passed = csvs
        result.label = result.label.str.casefold()
        return result

    def __from_inner__(self) -> Self:
        elsa = self.__outer__
        with self.configure:
            passed = self.passed
        result = (
            self
            .from_inferred(passed)
            .pipe(self)
            .indexed()
            .sort_index()
        )

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

        result.label = result.label.str.casefold()

        loc = result.isyn.isin(result.elsa.synonyms.drop_list.isyn)
        if loc.any():
            eg = (
                result.label
                .loc[loc]
                .unique()
                .tolist()
                [:10]
            )
            # warnings.warn(
            #     f'Dropping {loc.sum()} annotations from the truth '
            #     f'because they are in the drop_list e.g. {eg}'
            # )
            msg = (
                f'Dropping {loc.sum()} annotations from the truth '
                f'because they are in the drop_list e.g. {eg}'
            )
            self.__logger__.info(msg)
            result = result.loc[~loc].copy()

        if 'file' in result:
            result.file = util.trim_path(result.file)

        result.passed = passed
        # _ = result.ibox, result.isyn
        # loc = result.duplicated('ibox isyn'.split(), keep=False)
        # assert not loc.any()

        return result

    @magic.cached.property
    def unaccounted(self) -> Self:
        """Annotations that have not been accounted for in the images."""
        raise ValueError

    fake: Self

    # todo: false positives
    def fake(self) -> Self:
        """Use the Truth to generate some fake predictions"""
        s = self.norms.values
        w = self.normw.values
        n = self.normn.values
        e = self.norme.values

        def randomize_pair(lower: np.ndarray, upper: np.ndarray):
            skew = np.random.uniform(low=0.9, high=1.1)  # Same skew for both bounds of the pair
            new_lower = lower * skew
            new_upper = upper * skew
            new_lower, new_upper = np.clip(new_lower, 0, 1), np.clip(new_upper, 0, 1)
            # Ensure the relationship lower <= upper is maintained even after clipping
            new_lower, new_upper = np.minimum(new_lower, new_upper), np.maximum(new_lower, new_upper)
            return new_lower, new_upper

        # Apply the function to both pairs
        s, n = randomize_pair(s, n)
        w, e = randomize_pair(w, e)

        assert (s <= n).all()
        assert (w <= e).all()

        # 10% false negatives dropped
        shape = len(self)
        loc = np.ones(shape, dtype=bool)
        size = int(0.1 * shape)
        iloc = np.random.choice(shape, size=size, replace=False)
        loc[iloc] = 0

        # 10% false label positives
        label = self.label.values.copy()
        size = int(0.1 * len(self))
        iloc = np.arange(len(self))
        iloc = np.random.choice(iloc, size, replace=False)
        label[iloc] = np.random.choice(self.labels.label.values, size)

        # 10% false bounding box positives
        # to randomize, choose a random s and w [0, (image_with-width)]

        result = (
            self
            .__flush_columns__()
            .assign(norms=s, normw=w, normn=n, norme=e, label=label)
            .loc[loc]
            .pipe(self.__inner__)
        )
        return result

    # original: Self
    #
    # def original(self) -> Self:
    #     with self.configure:
    #         # noinspection PyUnresolvedReferences
    #         passed = self.__inner__.passed
    #     result = (
    #         self
    #         .from_inferred(passed)
    #         .indexed()
    #         .sort_index()
    #         # .pipe(self.__inner__)
    #         .pipe(self)
    #     )
    #     return result
    #
    @magic.column
    def irow(self):
        """The row from the CSV file that the data was located on."""


if __name__ == '__main__':
    truth = Truth.from_inferred(bing)
