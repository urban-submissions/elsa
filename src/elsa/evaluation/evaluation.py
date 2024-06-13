from __future__ import annotations

import numpy as np
from typing import *

from elsa.evaluation.grid import Grid

E = RecursionError, AttributeError
from pandas import Series
import pandas as pd
import geopandas as gpd
from functools import *
import magicpandas as magic
from pathlib import Path
from elsa.logits.logits import Logits

if False:
    from elsa.root import Elsa


def norm(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result: np.ndarray = func(*args, **kwargs)

        # round to 8 decimal places
        # tolerance = 1e-1
        tolerance = 1e-1
        loc = (np.abs(result - 1) < tolerance) | (np.abs(result) < tolerance)
        result = np.where(loc, np.round(result), result)

        loc = (result >= -tolerance) & (result <= 1.)
        assert loc.all(), 'result is not correctly normalized'

        return result

    return wrapper


def positive(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result: np.ndarray = func(*args, **kwargs)

        tolerance = -1e-8
        loc = result > tolerance
        loc &= result < 0
        result = np.where(loc, 0., result)

        loc = result >= 0
        assert loc.all(), 'result is not positive'
        return result

    return wrapper


class Evaluate(magic.Magic):
    __outer__: Elsa

    __call__: Grid

    def __call__(
            self,
            concatenated: str | Path,
            logits: Path | str = None,
            score: str = 'selected.loglse',
            threshold: float = .3,
            force=False,
    ) -> Grid:
        """
        checkpoint:
            Path to concatenated logits thresholded by score;
            if not found, will write to this path. Caution: with
            a very low threshold the file might be very large.
        logits:
            If checkpoint is to be constructed, this is the logits
            directory used
        score:
            score from logits.score.py to use when thresholding;
            choose from:
                whole.loglse
                whole.avglse
                whole.argmax
                selected.loglse
                selected.avglse
                selected.argmax
        threshold:
            threshold for selecting logits by score
        """
        elsa = self.__outer__
        concatenated = Path(concatenated)
        if (
                not concatenated.exists()
                or force
        ):
            if logits is None:
                raise ValueError(
                    'Checkpoint file not found. Please provide the '
                    'logits directory to initialize the checkpoint.'
                )
            logits = Path(logits)
            result = self.create_checkpoint(
                logits=logits,
                score=score,
                threshold=threshold,
            )
            result.to_parquet(concatenated)
        else:
            try:
                result = gpd.read_parquet(concatenated)
            except ValueError:
                result = pd.read_parquet(concatenated)
        result = result.pipe(Concatenated)
        if result.index.names != ['prompt', 'ilogit']:
            # todo something with Evaluation is breaking reset_index!!
            result = (
                result
                .pipe(pd.DataFrame)
                .reset_index()
                .set_index('prompt ilogit'.split(), append=False)
                .pipe(Concatenated)
            )
        # print(f'Checkpoint nfiles: {result.nfile.nunique()}')
        result.elsa = elsa
        c = result
        normwidth = c.normwidth
        normheight = c.normheight

        # todo: we shouldn't be doing this, it's a temporary fix
        loc = c.normx + normwidth / 2 >= 1
        c.loc[loc, 'normwidth'] = (1 - c.normx.loc[loc]) * 2
        loc = c.normy + normheight / 2 >= 1
        c.loc[loc, 'normheight'] = (1 - c.normy.loc[loc]) * 2
        loc = c.normx - normwidth / 2 < 0
        c.loc[loc, 'normwidth'] = c.normx.loc[loc] * 2
        loc = c.normy - normheight / 2 < 0
        c.loc[loc, 'normheight'] = c.normy.loc[loc] * 2
        loc = c.area != 0
        c = c.loc[loc].copy()
        _ = c.normw, c.normn, c.norme, c.norms, c.geometry
        grid = c.grid
        # grid = result.grid
        grid.using = score

        return grid


class Concatenated(Logits):
    # grid = Grid()
    colnames = (
        'prompt file ilogit score normx normy normwidth normheight'
    ).split()

    @Grid
    def grid(self) -> Grid:
        """
        Each ground truth box can match to multiple predictions.
        These mathces can be visualized as a grid
        """


    @magic.index
    def prompt(self) -> magic[str]:
        ...

    @magic.index
    def file(self) -> magic[str]:
        ...

    @magic.index
    def ilogit(self) -> magic[int]:
        ...

    @magic.column
    def ulogit(self) -> magic[int]:
        """unique integer for each (prompt, ilogit) pair"""
        test = self.reset_index()
        loc = test.duplicated('prompt ilogit'.split(), keep=False)
        # columns = [
        #     col
        #     for col in test.columns
        #     if col.startswith('scores.')
        # ]
        cols = 'prompt ilogit scores.selected.loglse'.split()
        test.loc[loc, cols].sort_values('prompt ilogit'.split())
        assert not (
            self
            .reset_index()
            .duplicated('prompt ilogit'.split())
            .any()
        )
        result = np.arange(len(self))
        return result

    @magic.column.from_options(no_recursion=True)
    def w(self) -> Series[float]:
        try:
            return self.x.values - self.width.values / 2
        except E:
            ...
        try:
            return self.image_width * self.normw.values

        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def s(self) -> Series[float]:
        try:
            return self.y.values - self.height.values / 2
        except E:
            ...
        try:
            return self.image_height * self.norms.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def e(self) -> Series[float]:
        try:
            return self.x.values + self.width.values / 2
        except E:
            ...
        try:
            return self.image_width * self.norme.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def n(self) -> Series[float]:
        try:
            return self.y.values + self.height.values / 2
        except E:
            ...
        try:
            return self.image_height * self.normn.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def x(self) -> Series[float]:
        try:
            return (self.w.values + self.e.values) / 2
        except E:
            ...
        try:
            return self.image_width * self.normx.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def y(self) -> Series[float]:
        try:
            return (self.s.values + self.n.values) / 2
        except E:
            ...
        try:
            return self.image_height.values * self.normy.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def height(self) -> Series[float]:
        try:
            return self.n.values - self.s.values
        except E:
            ...
        try:
            return self.image_height * self.normheight
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def width(self) -> Series[float]:
        try:
            return self.e.values - self.w.values
        except E:
            ...
        try:
            return self.image_width * self.normwidth
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def normw(self) -> Series[float]:
        try:
            return self.w.values / self.image_width.values
        except E:
            ...
        try:
            return self.normx.values - self.normwidth.values / 2
        except E:
            ...
        try:
            return self.norme.values - self.normwidth.values
        except E:
            ...
        try:
            return self.fw.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def norms(self) -> Series[float]:
        try:
            return self.s.values / self.image_height.values
        except E:
            ...
        try:
            return self.normy.values - self.normheight.values / 2
        except E:
            ...
        try:
            return self.normn.values - self.normheight.values
        except E:
            ...
        try:
            return self.fs.values - self.nfile
        except E:
            ...

        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def norme(self) -> Series[float]:
        try:
            return self.e.values / self.image_width.values
        except E:
            ...
        try:
            return self.normx.values + self.normwidth.values / 2
        except E:
            ...
        try:
            return self.normw.values + self.normwidth.values
        except E:
            ...
        try:
            return self.fe.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def normn(self) -> Series[float]:
        try:
            return self.n.values / self.image_height.values
        except E:
            ...
        try:
            return self.normy.values + self.normheight.values / 2
        except E:
            ...
        try:
            return self.norms.values + self.normheight.values
        except E:
            ...
        try:
            return self.fn.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def normx(self) -> Series[float]:
        try:
            return self.x.values / self.image_width.values
        except E:
            ...
        try:
            return (self.normw.values + self.norme.values) / 2
        except E:
            ...
        try:
            return self.fw.values - self.nfile
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def normy(self) -> Series[float]:
        try:
            return self.y.values / self.image_height.values
        except E:
            ...
        try:
            return (self.norms.values + self.normn.values) / 2
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def normwidth(self) -> Series[float]:
        try:
            return self.width.values / self.image_width.values
        except E:
            ...
        try:
            return self.norme.values - self.normw.values
        except E:
            ...
        raise AttributeError

    @magic.column.from_options(no_recursion=True)
    def normheight(self) -> Series[float]:
        try:
            return self.height.values / self.image_height.values
        except E:
            ...
        try:
            return self.normn.values - self.norms.values
        except E:
            ...
        raise AttributeError

    @classmethod
    def create_checkpoint(
            cls,
            elsa: Elsa,
            result: Path | str,
            score: str,
            threshold: float,
    ) -> Concatenated:
        """
        First selects all the logits from the directory above the
        threshold to make a single dataframe. This can be passed
        back to evaluate to avoid the long load times
        """
        it = Logits.from_directory(
            indir=result,
            score=score,
            threshold=threshold,
        )
        concat = list(it)
        result = (
            pd.concat(concat, axis=0)
            .pipe(Logits)
        )
        files = elsa.files
        result.elsa = elsa
        nfile = (
            files.nfile
            .set_axis(files.file)
            .loc[result.file]
            .values
        )
        ifile = (
            files
            .reset_index()
            .ifile
            .set_axis(files.file)
            .loc[result.file]
            .values
        )
        result: Logits = (
            result.assign(
                nfile=nfile,
                ifile=ifile,
            )
            .pipe(Concatenated)
        )
        result.elsa = elsa
        result.columns = result.columns.get_level_values(0)
        result.columns.name = None
        result.prompt = result.prompt.astype(dtype='category')
        result.file = result.file.astype(dtype='category')
        result: Concatenated = result.set_index('file prompt'.split())

        return result

    @cached_property
    def c(self) -> Self:
        loc = self.level == 'c'
        return self.loc[loc].copy()

    @cached_property
    def cs(self) -> Self:
        loc = self.level == 'cs'
        return self.loc[loc].copy()

    @cached_property
    def csa(self) -> Self:
        loc = self.level == 'csa'
        return self.loc[loc].copy()

    @magic.column
    def isyns(self):
        prompts = self.elsa.prompts
        result = (
            prompts.isyns
            .set_axis(prompts.natural)
            .loc[self.prompt]
            .values
        )
        return result
