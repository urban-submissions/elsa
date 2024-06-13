from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import json
import numpy as np
import pandas as pd
import shapely
import warnings
from functools import *
from geopandas import GeoSeries
from numpy import ndarray
from pandas import Series
from pathlib import Path
from shapely import *
from typing import *

import magicpandas as magic
from elsa import util
from elsa.resource import Resource

if False:
    from .root import Elsa

bing = Path(__file__, *'.. static bing images.csv'.split()).resolve()
google = Path(__file__, *'.. static google images.csv'.split()).resolve()
unified = {
    'BSV_': bing,
    'GSV_': google,
}
bing = Path(__file__, *'.. .. .. gt_data triple_inspected_May23rd bing images.csv'.split()).resolve()
google = Path(__file__, *'.. .. .. gt_data triple_inspected_May23rd google images.csv'.split()).resolve()
unified = {
    'BSV_': bing,
    'GSV_': google,
}

class Images(Resource):
    width: Series
    height: Series
    __owner__: Elsa
    __outer__: Elsa

    def __from_inner__(self) -> Self:
        """Called when accessing Elsa.images to instantiate Images"""
        with self.configure:
            passed = self.passed
        result = (
            self
            .from_inferred(passed)
            .indexed()
        )
        result.passed = passed
        loc = result.file.isin(result.duplicates.file)
        if loc.any():
            warnings.warn(
                f"Dropping {loc.sum()} duplicate images from "
                f"{result.passed}"
            )
            result = result.loc[~loc]
        return result


    def indexed(self):
        return self.set_index('ifile')

    @magic.column
    def width(self) -> magic[int]:
        """Width of the image in pixels"""
        path: Series[str] = self.path
        assert path.str.endswith('.png').all()
        def submit(path):
            with open(path, 'rb') as f:
                f.seek(16)  # PNG width is stored at byte 16-19
                width = int.from_bytes(f.read(4), 'big')
            return width
        with ThreadPoolExecutor() as threads:
            it = threads.map(submit, path)
            result = list(it)
        return result


    @magic.column
    def height(self) -> Series[int]:
        """Height of the image in pixels"""
        path: Series[str] = self.path
        assert path.str.endswith('.png').all()
        def submit(path):
            with open(path, 'rb') as f:
                f.seek(20)
                height = int.from_bytes(f.read(4), 'big')
            return height
        with ThreadPoolExecutor() as threads:
            it = threads.map(submit, path)
            result = list(it)
        return result

    @magic.column
    def path(self) -> Series[str]:
        """Path to the image file"""
        result = (
            self.files.path
            .loc[self.ifile]
            .values
        )
        return result

    @magic.index
    def source(self):
        """Source prefix used in ifile e.g. GSV_ for GSV_123"""
        return ''

    @classmethod
    def from_inferred(cls, path: str | Path) -> Self:
        if isinstance(path, dict):
            result = cls.from_paths(path)
        else:
            path = Path(path)
            match path.suffix:
                case '.json':
                    result = cls.from_json(path)
                case '.csv' | '.txt':
                    result = cls.from_csv(path)
                case _:
                    raise ValueError(f'Unsupported extension {path.suffix}')
        result.file = util.trim_path(result.file)
        result.passed = path
        return result

    @classmethod
    def from_paths(
            cls,
            paths: dict[str, str | Path],
    ) -> Self:
        """
        Construct Images from a dictionary of source prefixes to paths
        e.g. {'GSV_': 'path/to/gsv.csv', 'BSV_': 'path/to/bsv.csv'}"""
        concat = []
        for source, path in paths.items():
            result = cls.from_inferred(path)
            ifile = source + result.ifile.astype(str)
            result = result.assign(
                source=source,
                ifile=ifile,
            )
            concat.append(result)
        result = pd.concat(concat)
        result = result.pipe(cls)
        result.passed = paths
        return result

    @classmethod
    def from_csv(cls, path: str | Path) -> Self:
        """Construct Images from a CSV file."""
        result = (
            pd.read_csv(path)
            .pipe(cls)
        )
        result.passed = path
        return result

    @magic.cached.property
    def duplicates(self) -> Self:
        """The subset of files that have been dropped due to being duplicated."""
        loc = self.file.duplicated(keep=False)
        result = self.loc[loc]
        return result

    @magic.cached.property
    def where_empty(self) -> Self:
        """An image is empty if there are no ground truth annotations for it."""
        truth = self.truth
        if not truth.passed:
            raise ValueError('No truth has been passed')
        loc = ~self.file.isin(truth.file)
        result = self.loc[loc]
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> Self:
        """Construct Images from a JSON file."""
        with open(path) as file:
            data = json.load(file)
        images: list[dict[str, str | int]] = data['images']
        count = len(images)

        def fromiter(key: str, dtype=object) -> ndarray:
            return np.fromiter((
                img[key]
                for img in images
            ), dtype, count=count)

        width = fromiter('width', int)
        height = fromiter('height', int)
        ifile = fromiter('id', int)
        file = fromiter('file_name')
        result = cls({
            'width': width,
            'height': height,
            'ifile': ifile,
            'file': file,
        })
        result.passed = path
        result.file = result.file

        return result

    @classmethod
    def from_owner(cls, owner: Elsa) -> Self:
        """
        Instead of reading a file, just assign an ifile
        for each unique filename in the process.
        """
        raise NotImplementedError

    @magic.cached.cmdline.property
    def filter_empty(self) -> bool:
        """Filter out images with no ground truth annotations."""
        return True

    @magic.geo.column
    def geometry(self) -> GeoSeries[Polygon]:
        """A box around the image."""
        w = s = self.ifile.values
        e = n = s + 1
        data = shapely.box(w, s, e, n)
        result = GeoSeries(data, index=self.ndex)
        return result

    @cached_property
    def file2ifile(self) -> Series[str]:
        """Map file to ifile"""
        result = Series(self.ifile.values, index=self.file)
        return result

    @magic.column
    def nfile(self) -> Series[int]:
        """An integer [0, N] for each unique ifile"""
        ifile = self.ifile.unique()
        result = (
            Series(np.arange(len(ifile)), ifile)
            .loc[self.ifile]
            .values
        )
        return result


if __name__ == '__main__':
    images = Images.from_json(labels)
    images
