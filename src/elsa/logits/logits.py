from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

import numpy as np
import os
import pandas as pd
import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from functools import *
from pandas import DataFrame
from pandas import Index, MultiIndex
from pandas import Series
from pathlib import Path
from scipy.special import logsumexp
from typing import *

import magicpandas as magic
from elsa import boxes
from elsa import util
from elsa.logits.ranks import Ranks
from elsa.logits.scores import Scores

# from elsa.evaluation.evaluation
E = RecursionError, AttributeError

if False:
    from elsa.root import Elsa


class Maximums(magic.Frame):
    __outer__: Logits

    def __repr__(self):
        return super().__repr__()

    # @magic.column
    # def activity(self) -> magic[float]:
    #     """maximum confidence for any activity logit"""
    #
    # @magic.column
    # def state(self) -> magic[float]:
    #     """maximum confidence for any state logit"""
    #
    # @magic.column
    # def condition(self) -> magic[float]:
    #     """maximum confidence for any condition logit"""
    #
    # @magic.column
    # def others(self) -> magic[float]:
    #     """maximum confidence for any other logit"""

    @cached_property
    def labels(self) -> Self:
        loc = self.columns.isin(self.__outer__.label)
        result = self.loc[:, loc]
        return result

    def __from_inner__(self):
        logits = self.__outer__
        # use the maximum confidence for spans within a label
        label = (
            logits.confidence
            .T
            .groupby(level='label', sort=False)
            .max()
            .T
        )
        # use the maximum confidence for labels within a meta
        meta = (
            logits.confidence
            .T
            .groupby(level='meta', sort=False)
            .max()
            .T
        )
        concat = label, meta
        # noinspection PyTypeChecker
        result = (
            pd.concat(concat, axis=1)
            .pipe(self)
        )
        return result


class MaxLogit(magic.Frame):
    __outer__: Logits

    @magic.column
    def ifirst(self):
        ...

    @magic.column
    def min(self):
        ...

    @magic.column
    def max(self):
        ...

    def __from_inner__(self) -> Self:
        """
        for each box, we get the max logit and then the max and min
        of these max logits
        """
        logits = self.__outer__
        index = logits.confidence.columns.get_level_values('ifirst')
        ifirst = (
            logits.confidence
            .set_axis(index, axis=1)
            .idxmax(axis=1)
        )
        min = (
            logits.confidence
            .min(axis=0)
            .reset_index()
            .set_index('ifirst')
            [0]
            .loc[ifirst]
            .values
        )
        max = (
            logits.confidence
            .max(axis=0)
            .reset_index()
            .set_index('ifirst')
            [0]
            .loc[ifirst]
            .values
        )
        result = self({
            'ifirst': ifirst,
            'min': min,
            'max': max,
        }, index=logits.index)
        return result


class Logits(boxes.Base):
    columns: MultiIndex
    loc: MutableMapping[Any, Self] | Self
    iloc: MutableMapping[Any, Self] | Self
    max_logit = MaxLogit()
    scores = Scores()
    ranks = Ranks()
    without_extraneous_spans: Self
    without_irrelevant_files: Self

    @cached_property
    def confidence(self) -> Self:
        """Select only the columns that describe label confidence"""
        # loc = self.label != ''
        loc = self.ifirst != ''
        result = self.loc[:, loc]
        return result

    @magic.index
    def ilogit(self) -> magic[int]:
        """index of the logits"""

    def __repr__(self):
        return super().__repr__()

    # @cached_property
    # def meta(self) -> Index:
    #     """return the column names that describe the metalabel"""
    #     return self.columns.get_level_values('meta')
    #
    # @cached_property
    # def label(self) -> Index:
    #     """return the column names that describe the label"""
    #     return self.columns.get_level_values('label')

    @cached_property
    def span(self) -> Index:
        """return the column names that describe the span"""
        return self.columns.get_level_values('span')

    @cached_property
    def ifirst(self) -> Index:
        """return the column names that describe the start of the span"""
        return self.columns.get_level_values('ifirst')

    # @cached_property
    # def conditions(self) -> Self:
    #     """select only the columns that describe confidence of conditions"""
    #     loc = self.meta == 'condition'
    #     result = self.loc[:, loc]
    #     return result
    #
    # @cached_property
    # def cs(self) -> Self:
    #     loc = self.meta.isin(['condition', 'state'])
    #     result = self.loc[:, loc]
    #     return result
    #
    # @cached_property
    # def csa(self) -> Self:
    #     loc = self.meta.isin(['condition', 'state', 'activity'])
    #     result = self.loc[:, loc]
    #     return result

    # @cached_property
    # def states(self) -> Self:
    #     """select only the columns that describe confidence of states"""
    #     loc = self.meta == 'state'
    #     result = self.loc[:, loc]
    #     return result
    #
    # @cached_property
    # def activities(self) -> Self:
    #     """select only the columns that describe confidence of activities"""
    #     loc = self.meta == 'activity'
    #     result = self.loc[:, loc]
    #     return result
    #
    # @cached_property
    # def others(self) -> Self:
    #     """select only the columns that describe confidence of other"""
    #     loc = self.meta == 'other'
    #     result = self.loc[:, loc]
    #     return result

    @magic.column.from_options(dtype='category')
    def file(self) -> magic[str]:
        """filename"""

    @magic.column.from_options(dtype='category')
    def path(self) -> magic[str]:
        """path to image"""

    @magic.cached.property
    def passed(self) -> str:
        """the passed path to the logits parquet"""

    @magic.column
    def prompt(self) -> magic[str]:
        """the (synonymous) prompt that was used to generate the logits"""

    @classmethod
    def from_file(
            cls,
            file: str | Path,
            elsa: Optional[Elsa] = None,
            as_cls: bool = True,
    ) -> Self:
        """
        construct from a single file;
        elsa carries metadata
        """
        file = Path(file)
        result = (
            pd
            .read_parquet(file)
            # .pipe(cls)
        )
        if as_cls:
            result = cls(result)
        if '.' in result.columns[0]:
            # I was experiencing an issue with to_parquet with multi-level columns
            # so I compressed them into one-level a.b.c.d; this expands it back to [a, b, c, d]
            # names = 'meta label span ifirst'.split()
            names = 'span ifirst'.split()
            columns = (
                result.columns
                .to_frame()
                [0]
                .str.split('.')
                .pipe(pd.MultiIndex.from_tuples, names=names)
            )
            result.columns = columns

        result.passed = file
        if elsa is not None:
            # result.__outer__ = result.__owner__ = elsa
            result.elsa = elsa
            # someone else might have generated the logits, need to
            #   get the paths from the user's files
            try:
                result.path = (
                    elsa.files.path
                    .set_axis(elsa.files.file)
                    .loc[result.file]
                    .values
                )
            except KeyError:
                ...
        if 'prompt' not in result:
            result['prompt'] = file.stem
            result['prompt'] = result['prompt'].astype('category')


        if not result.index.name:
            result.index.name = 'ilogit'
        if 'meta' in result.columns.names:
            result.columns.droplevel('meta label'.split())
        if (
                isinstance(result.columns, pd.MultiIndex)
                and not result.columns.names[0]
        ):
            result.columns.names = ('span', 'ifirst')
        assert not result.ilogit.duplicated().any(), "ilogit must be unique"
        result['logit_file'] = str(file)

        return result

    @magic.column
    def lse(self) -> magic[float]:
        result = logsumexp(self.confidence, axis=1)
        return result

    @magic.column
    def score(self) -> magic[float]:
        """The score to be used in evaluating the logits"""
        return self.lse.values

    @cached_property
    def scored(self) -> Self:
        """
        Return a subframe with the score and essential columns;
        to be used in concatenation for the evaluation
        """
        _ = self.score
        columns = (
            'prompt file score '
            'normx normy normwidth normheight'
        ).split()
        result = self.loc[:, columns]
        return result


    def view(
            self,
            ilogit: Union[int, List[int]],
            score: str = 'selected.loglse',
            background='black',
            show_filename: bool = True,
    ) -> Image:
        if isinstance(ilogit, int):
            ilogit = [ilogit]

        image_paths = self.path.loc[ilogit]
        match image_paths.nunique():
            case 0:
                raise ValueError("No image paths found")
            case 1:
                ...
            case _:
                raise ValueError("All logits must belong to the same file")
        file = self.file.loc[ilogit].iloc[0]
        path = image_paths.iloc[0]
        image = Image.open(path).convert("RGBA")
        image_width, image_height = image.size

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        try:
            font = ImageFont.truetype("Courier New", 12)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSansMono", 12)
            except OSError:
                try:
                    font = ImageFont.truetype("LiberationMono", 12)
                except OSError:
                    try:
                        font = ImageFont.truetype('Arial', 12)
                    except OSError:
                        font = ImageFont.truetype('C:\\Windows\\Fonts\\arial.ttf', 12)

        x = self.normx.loc[ilogit] * image_width
        y = self.normy.loc[ilogit] * image_height
        width = self.normwidth.loc[ilogit] * image_width
        height = self.normheight.loc[ilogit] * image_height
        w = x - width / 2
        e = x + width / 2
        n = y - height / 2
        s = y + height / 2

        scored = self.get_score(score)
        also = {
            'whole.argmax': self.scores.whole.argmax,
            'whole.loglse': self.scores.whole.loglse,
            score: scored,
        }
        labels = (
            self.confidence
            .set_axis(self.confidence.span, axis=1)
            .assign(**also)
            .iloc[:, [-1, -2, -3, *range(len(self.confidence.span))]]
            .loc[ilogit]
            .round(2)
        )

        # Create a new image with extra space for text
        new_width = image_width + 1500
        text_color = 'white' if background == 'black' else 'black'

        new_image = Image.new('RGBA', (new_width, image_height), background)
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)

        # Print the prompt once at the top of the image
        prompt = self.prompt.loc[ilogit].iloc[0]
        y_offset = 10
        if show_filename:
            draw.text((image_width + 10, y_offset), f'{file}', fill=text_color, font=font)
            y_offset += 20
        draw.text((image_width + 10, y_offset), f'{score}', fill=text_color, font=font)
        y_offset += 20

        colors = util.colors[:len(ilogit)]

        boxes = pd.DataFrame({
            'w': w,
            'e': e,
            'n': n,
            's': s,
            'ilogit': ilogit,
            'color': colors[:len(ilogit)],
        })

        # Draw the bounding boxes
        it = zip(boxes.w, boxes.e, boxes.n, boxes.s, colors)
        for wi, ei, ni, si, color in it:
            draw.rectangle([wi, ni, ei, si], outline=color, width=3)
        labels: DataFrame

        index = 'ilogit'
        ilogit = labels.reset_index()['ilogit'].astype(str)
        width = [ilogit.str.len().max()]

        COLORS = ['white', 'white'] + util.colors[:len(ilogit)]
        if background == 'white':
            COLORS = ['black', 'black'] + util.colors[:len(ilogit)]

        rows = labels.iloc[:, :3].__repr__().split('\n')
        for row, color in zip(rows, COLORS):
            draw.text((image_width + 10, y_offset), row, fill=color, font=font)
            y_offset += 20
        y_offset += 10

        rows = labels.iloc[:, 3:].__repr__().split('\n')
        for row, color in zip(rows, COLORS):
            draw.text((image_width + 10, y_offset), row, fill=color, font=font)
            y_offset += 20

        return new_image

    def get_score(
            self,
            score: str = 'selected.loglse',
    ) -> Series[float]:
        scores = self.scores
        if score in scores.everything.columns:
            # get the score from the scores frame
            result = scores.everything[score]
        else:
            # get it from the spans
            loc = self.span == score
            result = self.loc[:, loc]
        return result

    def view_top(
            self,
            file: str = None,
            path: str = None,
            score: str = 'selected.loglse',
            n: int = 5,
            **kwargs,
    ) -> Image:
        scored = self.get_score(score)

        if (
            file is None
            and path is None
        ):
            loc = scored.idxmax()
            file = self.file.loc[loc]
        if path is not None:
            loc = self.path.values == path
        else:
            loc = self.file.values == file
        scored = scored[loc]
        ilogit = self.ilogit.values[loc]
        iloc = np.argsort(scored)[::-1]
        if n is not None:
            iloc = iloc[:n]
        ilogit = ilogit[iloc]
        ilogit = ilogit.tolist()
        result = self.view(
            ilogit=ilogit,
            score=score,
            **kwargs,
        )
        return result

    def view_tops(
            self,
            scores: tuple[Series[float]],
            path: str = None,
            top_n: int = 10,
    ):
        if path is None:
            path = self.path.iloc[0]
        images = [self.view_top(top_n=top_n, score=score, path=path) for score in scores]

        widths, heights = zip(*(i.size for i in images))
        if images[0].width > images[0].height:  # Wider than tall
            total_width = sum(widths)
            max_height = max(heights)
            new_image = Image.new('RGBA', (total_width, max_height))

            x_offset = 0
            for im in images:
                new_image.paste(im, (x_offset, 0))
                x_offset += im.width
        else:  # Taller than wide
            max_width = max(widths)
            total_height = sum(heights)
            new_image = Image.new('RGBA', (max_width, total_height))

            y_offset = 0
            for im in images:
                new_image.paste(im, (0, y_offset))
                y_offset += im.height

        return new_image

    # Example usage:
    # instance = YourClass()
    # top_images = instance.view_top(top_n=5)
    # for img in top_images:
    #     img.show()

    @magic.cached.frame.property
    def elsa(self) -> Elsa:
        """Optional passed Elsa instance which adds functionality"""
        raise ValueError(
            'To use this feature Logits.elsa must be assigned; '
            'please pass the elsa parameter during construction.'
        )
        # noinspection PyTypeChecker
        return

    @magic.cached.property
    def extraneous(self) -> set[str]:
        """A set of spans that should be dropped if they appear in the output"""
        result = (
            'a an on at ing with or the ed s to and up down including'
        ).split()
        result = set(result)
        return result

    def without_extraneous_spans(self) -> Self:
        """Logits without the extraneous spans e.g. a, an, to, the"""
        extraneous = self.extraneous
        loc = [
            not span
            or span not in extraneous
            for span in self.span
        ]
        result = self.loc[:, loc]
        return result

    def without_irrelevant_files(self) -> Self:
        """
        Logits without the files that don't actually contain the prompt
        according to the truth.
        """
        loc = ~self.is_irrelevant_file
        result = self.loc[loc]
        return result

    @magic.column
    def is_irrelevant_file(self) -> magic[bool]:
        """
        Whether the file is does not actually contain the prompt
        according to the truth
        """
        prompt = self.prompt
        files = self.elsa.files
        loc = files.implicated(prompt)
        file = files.file.loc[loc]
        loc = self.file.isin(file)
        result = ~loc
        return result

    @property
    def groupby_files(self) -> Iterator[Self]:
        yield from self.groupby(
            'file',
            as_index=False,
            sort=False,
        )

    @magic.column
    def is_prompt_in_file(self) -> magic[bool]:
        """Whether the file contains the exact prompt in its ground truth"""
        self.elsa.truth.combos.natural

    @magic.column
    def is_synonym_in_file(self) -> magic[bool]:
        """Whether the file contains a synonymous prompt in its ground truth"""
        elsa = self.elsa
        prompts = elsa.prompts
        prompt = self.prompt.iloc[0]
        isyn = (
            prompts.isyns
            .set_axis(prompts.natural)
            .loc[prompt]
            .iloc[0]
        )
        loc = elsa.truth.isyns == isyn
        file = elsa.file.loc[loc]
        loc = self.file.isin(file)
        return loc

    @classmethod
    def scored_to_directory(
            cls,
            indir: Path | str,
            outdir: Path | str,
            elsa: Elsa = None,
    ):
        """
        indir: directory of logits
        outdir: directory of scores
        """

        # each infile is indir/subdir/file.parquet
        # each outfile is outdir/subdir/file.parquet
        indir = Path(indir)
        inpaths = list(indir.rglob('*.parquet'))
        outpaths = [
            outdir / infile.relative_to(indir)
            for infile in inpaths
        ]

        with ThreadPoolExecutor() as threads:
            from_file = cls.from_file

            def it_logits():
                it: Iterator[Future] = (
                    threads.submit(from_file, infile, elsa=elsa)
                    for infile in inpaths
                )
                prev = next(it)
                while True:
                    try:
                        curr = next(it)
                    except StopIteration:
                        break
                    yield prev.result()
                    prev = curr
                yield prev.result()

            def it_logits():
                yield from (
                    from_file(infile, elsa=elsa)
                    for infile in inpaths
                )

            it = zip(it_logits(), outpaths)
            futures = []
            for logits, outfile in it:
                outfile.parent.mkdir(parents=True, exist_ok=True)
                scores = logits.scores
                ranks = logits.ranks
                if 'prompt' in logits:
                    prompt = logits.prompt.values
                else:
                    prompt = outfile.stem
                data = {
                    'file': logits.file.values,
                    'ilogit': logits.index.values,
                    'prompt': prompt,
                    'scores.whole.argmax': scores.whole.argmax.values,
                    'scores.whole.loglse': scores.whole.loglse.values,
                    'scores.whole.avglse': scores.whole.avglse.values,
                    'scores.selected.loglse': scores.selected.loglse.values,
                    'scores.selected.avglse': scores.selected.avglse.values,
                    'ranks.whole.argmax': ranks.whole.argmax.values,
                    'ranks.whole.loglse': ranks.whole.loglse.values,
                    'ranks.whole.avglse': ranks.whole.avglse.values,
                    'ranks.selected.loglse': ranks.selected.loglse.values,
                    'ranks.selected.avglse': ranks.selected.avglse.values,
                    'normx': logits.normx.values,
                    'normy': logits.normy.values,
                    'normwidth': logits.normwidth.values,
                    'normheight': logits.normheight.values,
                }
                frame = pd.DataFrame(data)
                frame.file = frame.file.astype('category')
                frame.prompt = frame.prompt.astype('category')
                future = threads.submit(frame.to_parquet, outfile)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

    @classmethod
    def from_directory(
            cls,
            indir: Path | str,
            score: str = 'selected.loglse',
            threshold: float = .3,
            elsa: Elsa = None,
    ) -> Iterator[Self]:
        """
        yields logits where the score is above the threshold

        prompt  file    ilogit    score   normx   normy   normwidth   normheight
        """
        indir = Path(indir)
        inpaths = list(indir.rglob('*.parquet'))
        assert len(inpaths)
        score = score.split('.')
        columns = (
            'prompt file normx normy normwidth normheight '
        ).split()

        with ThreadPoolExecutor() as threads:
            from_file = cls.from_file

            def it_logits():
                it: Iterator[Future] = (
                    threads.submit(from_file, infile, elsa=elsa)
                    for infile in inpaths
                )
                prev = next(it)
                while True:
                    try:
                        curr = next(it)
                    except StopIteration:
                        break
                    yield prev.result()
                    prev = curr
                yield prev.result()

            it = zip(it_logits(), inpaths)
            for logits, infile in tqdm.tqdm(it, total=len(inpaths)):
                logits = cls(logits)
                obj = logits.scores
                for s in score:
                    obj = getattr(obj, s)
                obj = obj.values
                loc = obj >= threshold
                if 'prompt' not in logits:
                    logits['prompt'] = infile.stem
                    logits['prompt'] = logits['prompt'].astype('category')
                argmax = logits.scores.whole.argmax.values
                axis = logits.columns.get_level_values(0)
                scores = logits.scores.everything
                scores.columns = [
                    'scores.' + col
                    for col in scores.columns
                ]

                logits = (
                    logits
                    .set_axis(axis, axis=1)
                    [columns]
                )

                concat = logits, scores
                result = pd.concat(concat, axis=1)

                yield (
                    result
                    .reset_index()
                    .assign(score=obj, argmax=argmax)
                    .loc[loc]
                )

    @classmethod
    def put_top_logits_in_a_new_directory(
            cls,
            indir: Path | str,
            outdir: Path | str,
            n: int = 50,
            force=False,
            elsa: Elsa = None,
    ) -> None:
        """
        keep = {}
        For each prompt,
            for each file,
                for each score,
                    add top n logits to keep
        """

        # todo: this should be parallel processed

        indir = Path(indir)
        inpaths = list(indir.rglob('*.parquet'))
        outpaths = [
            outdir / infile.relative_to(indir)
            for infile in inpaths
        ]
        assert len(inpaths)
        futures = []

        with ThreadPoolExecutor() as threads:
            from_file = cls.from_file

            def it_logits():
                it: Iterator[Future] = (
                    threads.submit(from_file, infile, elsa=elsa)
                    for infile in inpaths
                )
                prev = next(it)
                while True:
                    try:
                        curr = next(it)
                    except StopIteration:
                        break
                    yield prev.result()
                    prev = curr
                yield prev.result()

            iterable = zip(it_logits(), inpaths, outpaths)
            it = tqdm.tqdm(iterable, total=len(inpaths))
            for logits, infile, outfile in it:
                if (
                    not force
                    and outfile.exists()
                ):
                    continue
                iloc = (
                    logits.scores.everything
                    .assign(file=logits.file)
                    .set_index('file', append=True)
                    .stack(future_stack=True)
                    .reset_index(level=[1,2])
                    .groupby(['file', 'level_2'], observed=True)
                    [0]
                    .nlargest(n=n)
                    .index
                    .get_level_values(2)
                    .unique()
                )

                logits = logits.iloc[iloc].copy()
                logits.attrs = {}
                outfile.parent.mkdir(parents=True, exist_ok=True)
                future = threads.submit(pd.DataFrame.to_parquet, logits, outfile)
                futures.append(future)

            for future in as_completed(futures):
                future.result()


    @classmethod
    def put_top_logits_in_a_new_directory(
            cls,
            indir: Path | str,
            outdir: Path | str,
            n: int = 50,
            force=False,
            elsa=None,
    ) -> None:
        indir = Path(indir)
        outdir = Path(outdir)
        inpaths = list(indir.rglob('*.parquet'))
        outpaths = [outdir / infile.relative_to(indir) for infile in inpaths]
        assert len(inpaths)

        def process_file(infile, outfile):
            logits = cls.from_file(infile, elsa=elsa)
            if not force and outfile.exists():
                return
            iloc = (
                logits.scores.everything
                .assign(file=logits.file)
                .set_index('file', append=True)
                .stack(future_stack=True)
                .reset_index(level=[1, 2])
                .groupby(['file', 'level_2'], observed=True)[0]
                .nlargest(n=n)
                .index
                .get_level_values(2)
                .unique()
            )

            logits = logits.iloc[iloc].copy()
            logits.attrs = {}
            outfile.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.to_parquet(logits, outfile)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_file, infile, outfile) for infile, outfile in zip(inpaths, outpaths)]
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                future.result()

    @magic.column
    def level(self) -> magic[str]:
        if self.elsa is None:
            raise ValueError('elsa must not be None to determine level')
        result = (
            self.elsa.prompts.natural2level
            .loc[self.prompt]
            .values
        )
        return result
