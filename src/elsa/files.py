from __future__ import annotations

import glob
import numpy as np
import os
import pandas as pd
import warnings
from pandas import DataFrame
from pandas import Series
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa import util
from elsa.resource import Resource


class ClassProperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


class LocalFiles(util.LocalFiles):
    bing: str = dict(
        dhodcz2='/home/arstneio/Downloads/Archive/bing',
        marco='/scratch/datasets/sidewalk-ballet/label_1k/old/bing/images'
    )
    google: str = dict(
        dhodcz2='/home/arstneio/Downloads/Archive/google',
        marco='/scratch/datasets/sidewalk-ballet/label_1k/old/google/images'
    )

    @ClassProperty
    def unified(cls):
        return cls.bing, cls.google


class Files(Resource):
    # todo: should be magic cached property
    extension: str = 'png'

    @magic.column
    def width(self) -> Series[int]:
        """ Width of the image in pixels """
        return self.images.width.loc[self.ifile].values

    @magic.column
    def height(self) -> Series[int]:
        """ Height of the image in pixels """
        return self.images.height.loc[self.ifile].values

    @magic.cached.property
    def batch_max_width(self) -> int:
        """maximum width of all images in batch"""
        return int(self.width.max())

    @magic.cached.property
    def batch_max_height(self) -> int:
        """maximum height of all images in batch"""
        return int(self.height.max())

    @magic.cached.property
    def batch_width(self) -> int:
        """scaled width of all images in batch"""
        return int(self.width.mean())

    @magic.cached.property
    def batch_height(self) -> int:
        """scaled height of all images in batch"""
        return int(self.height.mean())

    @magic.column
    def width_over_batch_width(self) -> Series[float]:
        """width ratio of the image after transform"""
        return self.width / self.batch_width

    @magic.column
    def height_over_batch_height(self) -> Series[float]:
        """height ratio of the image after transform"""
        return self.height / self.batch_height

    @magic.column
    def ratio(self):
        return self.width / self.height

    @magic.cached.property
    def batch_ratio(self):
        return self.batch_width / self.batch_height

    @magic.column
    def path(self) -> Series[str]:
        """The absolute path to each image file"""

    @classmethod
    def from_directory(cls, directory: str | Path) -> Self:
        """Create a Files object from a directory of images"""
        path = os.path.join(directory, f'*.{cls.extension}')
        images = glob.glob(path)
        file = np.fromiter((
            file.rsplit(os.sep, maxsplit=1)[-1]
            for file in images
        ), dtype=object, count=len(images))
        result = cls({
            'path': images,
            'file': file,
        })
        result.passed = directory
        return result

    @classmethod
    def from_paths(cls, paths: list[str | Path]) -> Self:
        """Create a Files object from a list of paths"""
        concat = list(map(cls.from_inferred, paths))
        result = pd.concat(concat)
        result = result.pipe(cls)
        result.passed = paths
        return result

    @classmethod
    def from_inferred(cls, path) -> Self:
        """Create a Files object, inferring the procedure from the input"""
        if isinstance(path, (list, tuple)):
            result = cls.from_paths(path)
        elif isinstance(path, (Path, str)):
            result = cls.from_directory(path)
        else:
            msg = f'Elsa.files expected a Path, str, or Iterable, got {type(path)}'
            raise TypeError(msg)
        result.passed = path
        return result

    def __from_inner__(self) -> Self:
        """Called when accessing Elsa.files to instantiate Files"""
        with self.configure:
            passed = self.passed
        result = (
            self
            .from_inferred(passed)
            .pipe(self)
        )
        result.file = util.trim_path(result.file)
        loc = ~result.file.isin(self.__owner__.images.file)
        if loc.any():
            eg = result.file[loc].iloc[0]
            warnings.warn(
                f'{loc.sum()} files in {passed} e.g. {eg} are not '
                f'present in the images metadata. These files will '
                f'be ignored.'
            )
            result = result.loc[~loc].copy()
        _ = result.ifile
        result = result.set_index('ifile')

        return result

    @magic.column
    def nboxes(self) -> magic[int]:
        """How many boxes in the truth belong to this file"""
        result = (
            self.truth.combos
            .groupby('ifile')
            .size()
            .loc[self.ifile]
            .values
        )
        return result

    # todo but prompts is a resource
    def implicated(
            self,
            by: Union[DataFrame, Series, str, list]
    ) -> Series[bool]:
        """
        Determine which files are implicated by the farme (e.g. prompts)
        files.implicated_by(prompts)
        files.implicated_by((person walking, a person sitting on a chair))
        files.implicated_by(((0,5,7), (3,27,50))
        """
        if isinstance(by, (str, tuple)):
            by = [by]
        if isinstance(by, list):
            # select files that contain any of the prompts
            prompts = self.elsa.prompts
            loc = prompts.combo.isin(by)
            loc |= prompts.natural.isin(by)
            loc |= prompts.isyns.isin(by)
            isyns = prompts.isyns.loc[loc].values
            loc = self.truth.isyns.isin(isyns)
            file = self.truth.file.loc[loc].values
            loc = self.file.isin(file)
            loc |= self.file.isin(by)
            loc |= self.path.isin(by)
        elif 'file' in by:
            # select files that are implicated by the frame
            loc = self.file.isin(by.file)
        elif 'isyns' in by:
            # select files that contain synonyms to the frame
            truth = self.truth
            loc = truth.isyns.isin(by.isyns)
            file = truth.file.loc[loc].values
            loc = self.file.isin(file)
        else:
            raise NotImplementedError
        return loc

        # if hasattr(frame, 'file'):
        #     loc = self.file.isin(frame.file)
        # else:
        #     raise NotImplementedError
        # return loc
