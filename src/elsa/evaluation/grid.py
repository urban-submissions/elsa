from __future__ import annotations

import warnings

import tqdm

import networkx
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from functools import cached_property
from itertools import chain
from pandas import Series
from typing import *
from typing import Self
from typing import Union, List

import magicpandas as magic
from elsa import util
from elsa.boxes import Boxes
from elsa.evaluation import average_precision
from elsa.evaluation.invalid import Invalid
from elsa.evaluation.stacks import Stacks
from elsa.truth import Truth

if False:
    from .evaluation import Concatenated
    import elsa.evaluation.scored


class AveragePrecision(average_precision.AveragePrecision):
    __outer__: Grid

    def __from_inner__(self) -> Self:
        grid = self.__outer__
        scores = [
            col
            for col in grid.columns
            if col.startswith('scores.')
        ]
        concat = [
            grid
            .scored(score)
            .average_precision
            .assign(score=score.split('scores.')[1])
            .set_index('score', append=True)
            for score in scores
        ]
        result = (
            pd.concat(concat, axis=0)
            .sort_index()
        )
        return result


class MCAUPRC(magic.Frame):
    __outer__: Grid

    def __from_inner__(self) -> Self:
        grid = self.__outer__
        scores = [
            col
            for col in grid.columns
            if col.startswith('scores.')
        ]
        data = {}
        for score in tqdm.tqdm(scores):
            scored = grid.scored(score)
            mcauprc = scored.mcauprc
            name = score.split('scores.')[1]
            data[name] = mcauprc
        result = pd.DataFrame(data)
        return result


class Grid(Boxes):
    __outer__: Concatenated
    invalid = Invalid()
    threshold = .90
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]
    stacks = Stacks()

    @AveragePrecision
    def average_precision(self) -> AveragePrecision:
        """DataFrame encapsulating the AveragePrecision for each score"""

    @MCAUPRC
    def mcauprc(self) -> MCAUPRC:
        """DataFrame encapsulating the MCAUPRC for each score and box."""

    def __from_inner__(self) -> Self:
        eval: Concatenated = self.__outer__
        elsa = eval.elsa
        truth = TRUTH = elsa.truth.combos
        _ = truth.area, eval.area, truth.level
        checkpoint = LOGITS = eval

        # only match prompts that are being assessed
        needles = eval.prompt
        haystack = truth.prompts.natural
        loc = ~needles.isin(haystack.values)
        nunique = needles[loc].nunique()

        if nunique:
            total = needles.nunique()
            msg = (
                f'{nunique} evaluation prompts out of {total} do not exist '
                f'in the ground truth prompts. This means that {loc.sum()} '
                f'evaluation boxes out of {len(needles)} will be dropped. '
            )
            self.__logger__.info(msg)
            checkpoint = LOGITS = eval.loc[~loc]

        # match truth and  logits
        ILEFT, IRIGHT = util.sjoin(truth, checkpoint)
        ileft, iright = ILEFT, IRIGHT
        truth = TRUTH.iloc[ileft]
        checkpoint = LOGITS.iloc[iright]

        # select matches where IOU > threshold
        intersection = util.intersection(truth, checkpoint)
        union = util.union(truth, checkpoint)
        iou = intersection / union
        loc = iou >= self.threshold
        ileft, iright, iou = ileft[loc], iright[loc], iou[loc]
        truth, checkpoint = TRUTH.iloc[ileft], LOGITS.iloc[iright]
        intersection = intersection[loc]
        assert np.all(
            truth.file.astype(str).values == checkpoint.file.astype(str).values
        )

        if self.anchor:
            # choose logit that maximizes IOU with each truth
            iloc = (
                Series(iou)
                .groupby(ileft)
                .idxmax()
                .loc[ileft]
                .values
            )
            ianchor = iright[iloc]
            anchor = LOGITS.iloc[ianchor]
            # assert np.all(checkpoint.file.values == anchor.file.values)
            assert np.all(
                truth.file.astype(str).values == anchor.file.astype(str).values
            )

            # boxes in predictions that have more than 90% overlap with best
            area = util.intersection(anchor, checkpoint)
            intersection = area / checkpoint.area.values
            loc = intersection >= self.threshold
            iright = iright[loc]
            ileft = ileft[loc]
            ianchor = ianchor[loc]

            truth = TRUTH.iloc[ileft]
            checkpoint = LOGITS.iloc[iright]
            area = util.intersection(truth, checkpoint)
            intersection = area / truth.area.values
            uanchor = LOGITS.ulogit.values[ianchor]
            assert ~LOGITS.ulogit.duplicated().any()

        else:
            uanchor = -1

        _ = elsa.prompts.isyns
        isyns = (
            elsa.prompts
            .drop_duplicates('natural')
            .reset_index()
            .set_index('natural')
            .isyns
            .loc[checkpoint.prompt]
            .values
        )

        imatch = pd.Index(np.arange(len(checkpoint)), name='imatch')
        checkpoint = checkpoint.copy()
        _ = checkpoint.level, checkpoint.ulogit
        result: pd.DataFrame = (
            checkpoint
            .reset_index()
            .assign(
                uanchor=uanchor,
                isyns=isyns,
                intersection=intersection,
                ibox=truth.ibox.values,
            )
            .sort_values('ibox prompt'.split())
            .set_axis(imatch)
        )
        columns = 'ibox isyns'.split()
        columns += [col for col in result.columns if col not in columns]
        result = result[columns]
        return result

    @magic.cached.property
    def anchor(self) -> bool:
        return True

    scored: elsa.evaluation.scored.Scored

    def scored(
            self,
            score=None,
            cdba=False,
            subgroup_threshold: float = None,
    ) -> elsa.evaluation.scored.Scored:
        """
        Score the grid using the given score, which is a substring
        of the column to be used's name

        score:
            score from logits.score.py to use for analytics;
            default uses score from elsa.evaluate()

            choose from:
                whole.loglse
                whole.avglse
                whole.argmax
                selected.loglse
                selected.avglse
                selected.argmax

        cdba:
            True
                filter boxes that do not pass CDBA; see Scored.cdba

        subgroup_threshold:
            0
                do not subgroup
            float
                threshold for subgroups matched on the truth;
                predictions matched on a truth need to have this much
                IOU to be grouped for NMS; see Grid.subgroups;
        """
        self: Grid
        # todo: what if self.inner is to be a copy of self, but as a new class?
        result = self.__inner__(self)
        result.__trace__ = self.__trace__ + 'scored'
        if score is not None:
            result.using = score
        else:
            result.using = self.using

        del result.score
        _ = result.score
        result = result.sort_values('score', ascending=False)
        result.elsa = self.elsa

        if cdba:
            loc = result.cdba
            result = result.loc[loc].pipe(result)

        if subgroup_threshold is not None:
            result.subgroup_threshold = subgroup_threshold
        # cache superset so subsetting does not affect false negative count
        result.superset = result
        result.__trace__ = self.__trace__ + 'scored'
        result = self.__inner__(result)
        result.ibox
        return result

    @magic.cached.frame.property
    def superset(self) -> Self:
        """
        A cached instance of the Grid, so that grid.loc[...] still has
        access to the superset of the data. This is important for
        determining the fales negatives, which do not increase when
        we look at matches
        """

    @magic.index
    def imatch(self):
        ...

    @magic.index
    def ilogit(self):
        # ilogit is only unique per prompt per file
        ...

    @magic.index
    def ibox(self):
        ...

    @magic.column
    def file(self):
        ...

    @magic.column
    def nfile(self):
        ...

    @magic.column
    def intersection(self):
        ...

    @magic.column
    def argmax(self):
        ...

    @magic.column
    def normx(self):
        ...

    @magic.column
    def normy(self):
        ...

    @magic.column
    def normwidth(self):
        ...

    @magic.column
    def normheight(self):
        ...

    @magic.index
    def prompt(self):
        ...

    @magic.cached.property
    def using(self) -> str:
        """
        score from logits.score.py to use when thresholding;
        choose from:
            whole.loglse
            whole.avglse
            whole.argmax
            selected.loglse
            selected.avglse
            selected.argmax
        """

    @cached_property
    def elsa(self):
        return self.__outer__.elsa

    @magic.column
    def isyns(self):
        return (
            self.elsa.truth.combos.isyns
            .loc[self.ibox]
            .values
        )

    @magic.column
    def is_true_positive(self):
        """
        (1,2) == (1,2,3)?

        imatch  isyns   isyn    exists
        0       1,2,3   1       True
        0       1,2,3   2       True

        group by imatch, does all exist?
        """
        arrays = self.isyns, self.truth.isyns
        loc = pd.MultiIndex.from_arrays(arrays)
        result = (
            self.elsa.isyns.subcombo
            .stack()
            .loc[loc]
            .values
        )

        return result

    @magic.column
    def is_false_positive(self):
        return ~self.is_true_positive

    # todo: should be cached at scored instantiation,
    #   so subsetting scored dataframes does not affect false negative count

    # not in superset
    @magic.column
    def iou(self):
        intersection = util.intersection(self.truth, self)
        union = util.union(self.truth, self)
        result = intersection / union
        return result

    @cached_property
    def truth(self) -> Truth:
        """
        The ground truth combos DataFrame, aligned with the matches
        according to ibox.
        """
        return (
            self.elsa.truth.combos
            .loc[self.ibox]
        )

    @cached_property
    def scores(self) -> Self:
        loc = [
            col.startswith('scores.')
            for col in self.columns
        ]
        result = self.loc[:, loc]
        return result

    def includes(
            self,
            label: str = None,
            meta: str = None,
    ) -> Series[bool]:
        result = (
            self.stacks
            .includes(label, meta)
            .loc[self.ibox]
            .set_axis(self.index)
        )
        return result

    def excludes(
            self,
            label: str = None,
            meta: str = None,
    ) -> magic[bool]:
        result = ~self.includes(label, meta)
        return result

    def get_nunique_labels(self, loc=None) -> Series[bool]:
        result = (
            self.stacks
            .get_nunique_labels(loc)
            .loc[self.ibox]
            .set_axis(self.index)
        )
        return result

    @magic.series
    def is_invalid(self) -> Series[bool]:
        """
        Whether the boxes are invalid.
        invalid boxes are considered false positives.
        """
        invalid = self.invalid
        for check in invalid.checks:
            getattr(invalid, check)
        checks = list(invalid.checks)
        result = (
            invalid
            .loc[self.ibox, checks]
            .any(axis=1)
            .set_axis(self.index)
        )
        return result

    @magic.column
    def truth_isyns(self) -> magic[tuple[int]]:
        result = (
            self.elsa.truth.combos.isyns
            .loc[self.ibox]
            .values
        )
        return result

    def subgroups(self, threshold: float) -> Series[int]:
        """
        Within each group on ibox, create subgroups based on which
        predictions intresect each other.

        For example, if there are 4 prediction that intersect the truth,
        with two overlapping on the left half, and two overlapping on
        the right, they might be grouped together on the truth, but not
        truly spatially related.
        """
        sub = (
            self
            .reset_index()
            .assign(iloc=np.arange(len(self)))
            ['imatch iloc ibox n w s e'.split()]
        )
        merge = sub.merge(sub, on='ibox', suffixes='_l _r'.split(), how='inner')
        ileft = merge.iloc_l.values
        iright = merge.iloc_r.values
        left = sub.iloc[ileft]
        right = sub.iloc[iright]

        intersection = util.intersection(left, right)
        union = util.union(left, right)
        iou = intersection / union
        loc = iou >= threshold
        ileft = ileft[loc]
        iright = iright[loc]

        g = networkx.Graph()
        edges = np.c_[ileft, iright]
        g.add_edges_from(edges)
        cc = list(networkx.connected_components(g))

        nold = self.ibox.nunique()
        nnew = len(cc)
        assert nnew >= nold, f'{nnew} < {nold}'

        repeat = np.fromiter(map(len, cc), int, len(cc))
        igroup = np.repeat(np.arange(len(cc)), repeat)
        it = chain.from_iterable(cc)
        iloc = np.fromiter(it, int, repeat.sum())
        imatch = self.imatch.values[iloc]
        result = (
            Series(igroup, index=imatch)
            .loc[self.imatch]
            .values
        )
        return result

    @magic.cached.property
    def subgroup_threshold(self):
        return .9

    @magic.series
    def igroup(self) -> magic[int]:
        """Subgroup computed by Grid.subgroups to be used for NMS"""
        result = self.subgroups(self.subgroup_threshold)
        return result

    @magic.column
    def ulogit(self):
        """
        Unique index for each logit;
        ilogit is only unique per prompt per file
        """

    @magic.column
    def uanchor(self):
        """
        ulogit of the logit which anchored the group to the truth box.
        """

    @magic.column
    def imatch_anchor(self):
        """imatch of the logit which maximized intersection with the truth for that box"""

    @magic.column
    def score(self):
        using = self.using
        cols = [
            col
            for col in self.columns
            if using in col
        ]
        if len(cols) == 0:
            msg = f'No columns found for score {using}'
            raise ValueError(msg)
        if len(cols) > 1:
            msg = f'Multiple columns found for score {using}'
            raise ValueError(msg)
        self.__logger__.info('Using score %s', using)
        SCORE = cols[0]
        result = self[SCORE]
        return result

    # @magic.Frame
    # def input(self):
    #     """
    #     Generate a subcombo matrix, where a 1 represents
    #     which combos the logit is a subcombo of.
    #     """
    #     result = (
    #         self.elsa.isyns.subcombo
    #         .astype(int)
    #         .loc[self.isyns]
    #     )
    #     return result
    #
    # @cached_property
    # def target(self) -> magic[bool]:
    #     """
    #     Get the integer location in the subcombo matrix
    #     of the combo that corresponds to the truth box
    #     """
    #     result = (
    #         self.elsa.isyns.subcombo.index
    #         .get_indexer(self.truth.isyns)
    #     )
    #     return result

    @magic.column
    def input(self):
        """
        Generate a subcombo matrix, where a 1 represents
        which combos the logit is a subcombo of.
        """
        result = self.elsa.isyns.subcombo.index.get_indexer(self.isyns)
        return result


    @magic.column
    def target(self) -> magic[bool]:
        """
        Each class is composed of labels, however average_precision
        is meant for individual labels. For this metric we determine
        which predictions are subcombos of the truth combos, and
        change the target combo to the prediction combo. This allows us
        to use integer ID representations.
        """
        subcombo = self.elsa.isyns.subcombo
        result = subcombo.index.get_indexer(self.truth.isyns)
        arrays = self.isyns, self.truth.isyns
        loc = pd.MultiIndex.from_arrays(arrays)
        is_subcombo = subcombo.stack().loc[loc].values
        isyns = self.isyns.loc[is_subcombo]
        result[is_subcombo] = subcombo.index.get_indexer(isyns)

        return result

    @property
    def nfiles(self):
        return self.file.nunique()

    def view(
            self,
            imatch: Union[int, List[int]],
            all_truth=False,
            file: str = None,
            background='black',
    ) -> Image:
        colors: list[str] = util.colors
        assert not self.imatch.duplicated().any()

        if isinstance(imatch, int):
            imatch = [imatch]

        iloc = self.imatch.get_indexer(imatch)
        assert len(set(self.iloc[iloc].file)) <= 1, "All imatches must belong to the same file"

        truth = self.truth
        logits = self
        t = truth
        l = logits
        _ = t.w, t.s, t.e, t.n, l.w, l.s, l.e, l.n, l.path, l.score

        truth = truth.iloc[iloc]
        logits = logits.iloc[iloc]
        names = self.index.names


        if file is None:
            file = logits.file.iloc[0]
            path = logits.path.iloc[0]
        else:
            path = (
                self.elsa.files
                .set_index('file')
                .path
                .xs(file)
            )

        image = Image.open(path).convert("RGBA")

        # Create a new image with extra space for text
        width, height = image.size
        new_width = width + 300
        new_image = Image.new('RGBA', (new_width, height), 'white' if background == 'white' else 'black')
        new_image.paste(image, (0, 0))

        draw = ImageDraw.Draw(new_image)
        font = ImageFont.load_default()

        header_text_color = 'black' if background == 'white' else 'white'

        util.draw_text_with_outline(draw, (width + 10, 10), f'file={file}', font, header_text_color, 'black', 1)
        util.draw_text_with_outline(draw, (width + 10, 30), f'anchor=yellow; truth=black', font, header_text_color, 'black', 1)

        try:
            anchor = (
                self
                .reset_index()
                .set_index('ulogit')
                .loc[logits.uanchor]
                .reset_index()
                .set_index(names)
            )
        except Exception:
            ...
        else:
            for ai in anchor.itertuples():
                xy = ai.w, ai.s, ai.e, ai.n
                draw.rectangle(xy, outline="white", width=3)

        y_offset = 50
        it = enumerate(zip(logits.itertuples(), truth.itertuples()))
        for i, (li, ti) in it:
            xy = ti.w, ti.s, ti.e, ti.n
            draw.rectangle(xy, outline="yellow", width=3)

        if all_truth:
            loc = self.elsa.truth.file == file
            it = self.elsa.truth.loc[loc].itertuples()
            for ti in it:
                xy = ti.w, ti.s, ti.e, ti.n
                draw.rectangle(xy, outline="yellow", width=3)

        it = enumerate(zip(logits.itertuples(), truth.itertuples()))
        for i, (li, ti) in it:
            color = colors[i % len(colors)]
            score = li.score

            xy = li.w, li.s, li.e, li.n
            draw.rectangle(xy, outline=color, width=3)

            text_score = f'score={score:.4f} imatch={li.Index}'
            util.draw_text_with_outline(draw, (width + 10, y_offset), text_score, font, color, 'black', 1)
            y_offset += 20

            text_truth = f'truth={ti.label}'
            util.draw_text_with_outline(draw, (width + 10, y_offset), text_truth, font, color, 'black', 1)
            y_offset += 20

            text_logit = f'logit={li.prompt}'
            util.draw_text_with_outline(draw, (width + 10, y_offset), text_logit, font, color, 'black', 1)
            y_offset += 20

        return new_image

    def view_top(
            self,
            file: str,
            n: int = 5,
            unique=False,
            all_truth=False,
    ) -> Image:
        """
        View top-scoring predictions for a file.

        file:
            str:    file to view
        n:
            int:    number of predictions to view
        unique:
            True:   see one prediction per truth box
        all_truth:
            True:   view all truth boxes
        """
        loc = self.file == file
        _ = self.score

        logits = (
            self
            .loc[loc]
            .sort_values('score', ascending=False)
        )
        if unique:
            imatch = (
                logits
                .reset_index()
                .groupby(logits.ibox.values)
                .imatch
                .first()
                .tolist()
            )
        else:
            imatch = (
                logits.imatch
                .tolist()
            )
        if n is not None:
            imatch = imatch[:n]

        result = self.view(
            imatch=imatch,
            all_truth=all_truth,
            file=file,
        )
        return result

    @cached_property
    def grid(self) -> Grid:
        return self

    @magic.column
    def level(self):
        ...

    @cached_property
    def c(self) -> Self:
        loc = self.level == 'c'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def cs(self) -> Self:
        loc = self.level == 'cs'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def csa(self) -> Self:
        loc = self.level == 'csa'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def cso(self) -> Self:
        loc = self.level == 'cso'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def csao(self) -> Self:
        loc = self.level == 'csao'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def person(self) -> Self:
        loc = self.condition == 'person'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def pair(self) -> Self:
        loc = self.condition == 'pair'
        result = self.loc[loc].copy()
        return result

    @cached_property
    def people(self) -> Self:
        loc = self.condition == 'people'
        result = self.loc[loc].copy()
        return result

    @magic.series
    def f1(self) -> Series[float]:
        def f1(sub) -> float:
            c = sub.precision * sub.recall * 2
            if c != 0:
                c /= sub.precision + sub.recall
            else:
                c = np.nan
            return c

        c = f1(self.c)
        cs = f1(self.cs)
        csa = f1(self.csa)
        cso = f1(self.cso)
        csao = f1(self.csao)
        person = f1(self.person)
        pair = f1(self.pair)
        people = f1(self.people)
        overall = f1(self)

        result = {
            'c': c,
            'cs': cs,
            'csa': csa,
            'cso': cso,
            'csao': csao,
            'person': person,
            'pair': pair,
            'people': people,
            'overall': overall,
        }
        result = Series(result)
        return result

    @cached_property
    def precision(
            self,
    ) -> Series[float]:
        """True Positives / (True Positives + False Positives)"""
        _ = self.ibox
        tp = self.is_true_positive.sum()
        fp = self.is_false_positive.sum()
        if tp + fp == 0:
            result = np.nan
        else:
            result = tp / (tp + fp)
        return result

    @cached_property
    def recall(
            self,
    ) -> Series[float]:
        """True Positives / (True Positives + False Negatives)"""
        tp = self.is_true_positive.sum()
        fn = self.false_negatives.sum()
        result = tp / (tp + fn)
        return result

    @cached_property
    def false_negatives(self) -> Series[int]:
        """Ground Truth Annotations which have not been matched"""
        truth = self.elsa.truth.combos
        loc = ~truth.ibox.isin(self.ibox)
        return loc


    @magic.column
    def condition(self) -> magic[str]:
        prompts = self.elsa.prompts
        _ = prompts.condition, prompts.natural
        result = (
            prompts
            .reset_index()
            .drop_duplicates('isyns')
            .set_index('isyns')
            .condition
            .loc[self.isyns]
            .values
        )
        return result


