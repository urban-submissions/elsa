from __future__ import annotations

import functools
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import *

import magicpandas as magic
from elsa.annotation.prompts import Prompts
from elsa.evaluation.evaluation import Concatenated
from elsa.evaluation.grid import Grid
from elsa.files import Files
from elsa.images import Images
from elsa.invalid import Invalid
from elsa.isyns import ISyns
from elsa.labels import Labels
from elsa.resource import Resource
from elsa.synonyms.synonyms import Synonyms
from elsa.truth.truth import Truth
from elsa.generate.predict import predict
from elsa.predictv3.predict import GDino3P


class Elsa(Resource):
    predict = predict
    gdino = GDino3P()

    @Images
    def images(self) -> Images:
        """
        A DataFrame of all Images, mapping filenames to paths,
        image sizes, and other metadata.
        """

    @Labels
    def labels(self) -> Labels:
        """
        A DataFrame of the unique Labels used by the dataset, mapping
        their names to their IDs,
        """

    @Truth
    def truth(self) -> Truth:
        """
        A DataFrame encapsulating the ground truth annotations from the
        dataset, containing the bounding boxes and their assigned labels.
        """

    @Files
    def files(self) -> Files:
        """
        A DataFrame representing the literal image files available;
        elsa.images contains metadata about the images but elsa.files
        contains all the files in the directory and their actual paths.
        """

    @Synonyms
    def synonyms(self) -> Synonyms:
        """
        A DataFrame representing which labels are synonymous, and other
        metadata such as their metalabel (condition, state, activity,
        other), natural representation (person -> a person), and whether
        these synonyms are used in the prompt generation.
        """

    @ISyns
    def isyns(self) -> ISyns:
        """
        A DataFrame representing all unique combinations of labels
        implied by the ground truth annotations.
        """

    @Invalid
    def invalid(self) -> Invalid:
        """
        A DataFrame representing all possible reasons that any
        annotation combinations in the dataset may be invalid. For
        example, "person standing sitting" is invalid: a person cannot
        be both standing and sitting.
        """



    evaluate: Grid
    def evaluate(
            self,
            concatenated: str | Path,
            logits: Path | str = None,
            score: str = 'selected.loglse',
            threshold: float = .3,
            force=False,
            anchor=False,
    ) -> Grid:
        # todo: maybe move this method to the Grid class
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
            result = Concatenated.create_checkpoint(
                self,
                result=logits,
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
        # todo: checkpoint.grid
        if result.index.names != ['prompt', 'ilogit']:
            # todo something with Evaluation is breaking reset_index!!
            result = (
                result
                .pipe(pd.DataFrame)
                .reset_index()
                .set_index('prompt ilogit'.split(), append=False)
                .pipe(Concatenated)
            )
        check = result
        check.elsa = self

        loc = check.ifile.isin(self.truth.ifile)
        ifile = check.ifile.loc[~loc]
        msg = (
            f'{ifile.nunique()} unique image files predicted for out of '
            f'{check.files.ifile.nunique()} are not included in the truth. '
        )
        self.__logger__.info(msg)
        check = check.loc[loc].copy()

        normwidth = check.normwidth
        normheight = check.normheight

        # todo: we shouldn't be doing this, it's a temporary fix
        loc = check.normx + normwidth / 2 >= 1
        check.loc[loc, 'normwidth'] = (1 - check.normx.loc[loc]) * 2
        loc = check.normy + normheight / 2 >= 1
        check.loc[loc, 'normheight'] = (1 - check.normy.loc[loc]) * 2
        loc = check.normx - normwidth / 2 < 0
        check.loc[loc, 'normwidth'] = check.normx.loc[loc] * 2
        loc = check.normy - normheight / 2 < 0
        check.loc[loc, 'normheight'] = check.normy.loc[loc] * 2
        loc = check.area != 0
        check = check.loc[loc].copy()
        _ = check.normw, check.normn, check.norme, check.norms, check.geometry
        if 'prompt' in check.columns:
            if 'prompt' not in check.index.names:
                raise NotImplementedError
            else:
                del check['prompt']
        if 'ilogit' in check.columns:
            if 'ilogit' not in check.index.names:
                raise NotImplementedError
            else:
                del check['ilogit']
        with check.configure:
            check.grid.anchor = anchor
        grid = check.grid
        grid.using = score

        # todo: make note of this, or point out this for the future;
        #   here we are calling self.inner(grid) to propagate self.trace
        #   and then grid(...) to propagate back the attributes;
        #   maybe implement something like self.carryover(...) to
        #   support when trace is different
        # result = grid.copy()
        # trace = self.__inner__(grid).__trace__
        # result.__trace__ = trace
        # todo: this doesn't work either. let's just manually assign for now
        #   and figure out later how to support inheritance of attributes
        # result = self.__inner__(grid)
        # trace = self.__inner__(grid).__trace__
        # temporary fix...
        result = grid.copy()
        result.__trace__ = self.__trace__ + 'evaluate'
        result.using = grid.using
        result.elsa = self
        return result

    with magic.default:
        default = magic.default(
            images=images.passed,
            labels=labels.passed,
            # predictions=predictions.passed,
            truth=truth.passed,
            files=files.passed,
        )


    @classmethod
    @default
    def from_resources(
            cls,
            truth=None,
            images=None,
            labels=None,
            files=None,
            original=None,
    ) -> Self:
        """
        Generate a Raster instance from the specified resource paths.

        truth:
            str | Path:
                file or directory used to generate elsa.truth
        images:
            str | Path:
                file used to generate elsa.images
            dict[str, str | Path]:
                mapping of source name to image metadata file, e.g:
                {'BSV_': ..., 'GSV_': ...}
        labels:
            str | Path:
                file used to generate elsa.labels
        """
        _images = Images.from_inferred(images)
        _truth = Truth.from_inferred(truth)
        _files = Files.from_inferred(files)
        file2ifile = _images.file2ifile
        if 'ifile' not in _truth:
            first = file2ifile.loc[_truth.file]
        else:
            first = _truth.ifile
        if 'ifile' not in _images:
            second = file2ifile.loc[_images.file]
        else:
            second = _images.ifile
        concat = first, second
        index = (
            pd.concat(concat, ignore_index=True)
            .pipe(pd.Index)
            .unique()
        )
        result = cls(index=index)
        with result.configure:
            result.truth.passed = truth
            result.images.passed = images
            result.labels.passed = labels
            result.files.passed = files
            # result.truth.original.passed = original

        # drop files not in images or images not in files
        images = result.images
        files = result.files
        ifile = images.ifile.intersection(files.ifile)

        # drop images not in files
        loc = images.ifile.isin(ifile)
        total = len(images)
        dropped = (~loc).sum()
        if dropped:
            eg = (
                images.ifile
                [~loc]
                # .drop_duplicates()
                # .tolist()
                # [:10]
            )
            msg = (
                f'{dropped} files e.g. {eg} in the images metadata out '
                f'of {total} are not in the literal image files and are '
                f'being dropped.'
            )
            result.__logger__.info(msg)
            images = images.loc[loc]

        # drop files not in images
        loc = files.ifile.isin(ifile)
        total = len(files)
        dropped = (~loc).sum()
        if dropped:
            eg = (
                files.ifile
                .loc[~loc]
                .drop_duplicates()
                .tolist()
                [:10]
            )
            msg = (
                f'{dropped} files e.g. {eg} in the files metadata out of'
                f' {total} are not in the images metadata and are being '
                f'dropped.'
            )
            result.__logger__.info(msg)
            files = files.loc[loc]

        result.files = files
        result.images = images

        # drop truth not in files
        truth = result.truth
        files = result.files
        ifile = files.ifile.intersection(truth.ifile)
        loc = truth.ifile.isin(ifile)
        total = truth.ifile.nunique()
        dropped = truth.ifile.loc[~loc].nunique()
        if dropped:
            eg = (
                truth.ifile
                # truth.
                .loc[~loc]
                .drop_duplicates()
                .tolist()
                [:10]
            )
            msg = (
                f'{dropped} files e.g. {eg} in the truth metadata out of'
                f' {total} are not in the files metadata and are being '
                f'dropped.'
            )
            truth = truth.loc[loc]
            result.__logger__.info(msg)

        result.truth = truth

        return result

    @classmethod
    def from_google(
            cls,
            images=None,
            labels=None,
            predictions=None,
            truth=None,
            files=None,
            original=None,
    ) -> Self:
        """Generate a Raster specific to the Google dataset."""
        if images is None:
            from elsa.images import google as images
        if labels is None:
            from elsa.labels import google as labels
        if predictions is None:
            from elsa.predictions import google as predictions
        if truth is None:
            from elsa.truth.truth import google as truth
        if original is None:
            from elsa.truth.truth import google as original
        if files is None:
            from elsa.files import LocalFiles
            files = LocalFiles.google
        result = cls.from_resources(
            images=images,
            labels=labels,
            predictions=predictions,
            truth=truth,
            files=files,
            original=original,
        )
        return result

    @classmethod
    def from_bing(
            cls,
            images=None,
            labels=None,
            predictions=None,
            truth=None,
            files=None,
            original=None,
    ) -> Self:
        """Generate a Raster specific to the Bing dataset."""
        if images is None:
            from elsa.images import bing as images
        if labels is None:
            from elsa.labels import bing as labels
        if predictions is None:
            from elsa.predictions import bing as predictions
        if truth is None:
            from elsa.truth.truth import bing as truth
        if original is None:
            from elsa.truth.truth import bing as original
        if files is None:
            from elsa.files import LocalFiles
            files = LocalFiles.bing
        result = cls.from_resources(
            images=images,
            labels=labels,
            predictions=predictions,
            truth=truth,
            files=files,
            original=original,
        )
        return result

    @classmethod
    def from_unified(
            cls,
            images=None,
            labels=None,
            predictions=None,
            truth=None,
            files=None,
            original=None,
    ) -> Self:
        """Generate a Raster that includes both Google and Bing datasets."""
        if images is None:
            from elsa.images import unified as images
        if labels is None:
            from elsa.labels import unified as labels
        if predictions is None:
            from elsa.predictions import unified as predictions
        if truth is None:
            from elsa.truth.truth import unified as truth
        if original is None:
            from elsa.truth.truth import original
        if files is None:
            from elsa.files import LocalFiles
            files = LocalFiles.unified
        result = cls.from_resources(
            images=images,
            labels=labels,
            predictions=predictions,
            truth=truth,
            files=files,
            original=original,
        )
        return result

    @functools.cached_property
    def prompts(self) -> Prompts:
        """
        a DataFrame of unique prompts that are to be used in the prediction.
        """
        consumed = self.truth.unique.rephrase.consumed
        result = consumed.prompts
        natural = result.natural
        isyns = result.index.get_level_values('isyns')
        appendix = (
            self.truth.unique.consumed.alone_appendix
            .loc[isyns]
            .set_axis(natural.index, axis=0)
        )
        result.natural = result.natural + appendix
        loc = ~result.natural.duplicated()
        result = result.loc[loc].copy()
        return result

    @property
    def file(self):
        raise NotImplementedError


