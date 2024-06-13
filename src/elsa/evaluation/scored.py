from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from functools import cached_property
from pandas import Series
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
from typing import *
from typing import Self

import magicpandas as magic
from elsa import util
from elsa.evaluation.grid import Grid
from elsa.evaluation.average_precision import (
    AveragePrecision,
)

if False:
    from elsa import Elsa
E = RecursionError, AttributeError


def compute_average_precision(
        precisions: np.ndarray,
        recalls: np.ndarray
) -> float:
    # COPYPASTED FROM LLM

    # Sort by recalls in ascending order
    sorted_indices = np.argsort(recalls)
    precisions = precisions[sorted_indices]
    recalls = recalls[sorted_indices]

    # Initialize average precision
    average_precision = 0.0

    # Compute average precision using the trapezoidal rule
    for i in range(1, len(recalls)):
        delta_recall = recalls[i] - recalls[i - 1]
        average_precision += precisions[i] * delta_recall

    return average_precision


def non_maximum_suppression(
        grid: Grid,
        threshold: float,
) -> np.ndarray:
    """Returns the subset of imatch that are not suppressed by NMS"""

    _ = grid.normw, grid.norms, grid.norme, grid.normn, grid.imatch
    boxes = grid['normw norms norme normn'.split()].values
    scores = grid['score'].values
    imatch = grid.imatch.values

    if len(boxes) == 0:
        return np.array([])

    boxes = np.array(boxes)
    scores = np.array(scores)
    imatch = np.array(imatch)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    selected_imatch = imatch[keep]
    return selected_imatch


class Scored(Grid):
    loc: MutableMapping[Any, Self]
    iloc: MutableMapping[Any, Self]
    average_precision = AveragePrecision()

    @magic.column
    def nms(self) -> magic[bool]:
        """True if the prediction is not suppressed by NMS"""
        _ = self.normw, self.normn, self.norme, self.norms, self.igroup
        # todo: nms can probably be done without apply
        imatch = np.concatenate(
            self
            .groupby(self.igroup.values)
            .apply(non_maximum_suppression, threshold=self.threshold)
            .values
        )
        result = self.imatch.isin(imatch)
        return result

    #
    @cached_property
    def ap(self) -> float:
        """Average Precision"""
        # todo: compute average precision from these matrices

    @cached_property
    def ap_torch(self):
        raise NotImplementedError(
            f''
            f'Should be implemented with classes as input/target labels,'
            f' not ibox.'
        )
        # self = self.copy()
        iou_thresholds = [0.75, 0.8, 0.85, 0.9]
        metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=iou_thresholds,
            max_detection_thresholds=[100, 1000, 1000]
        )
        target = []
        preds = []
        size = len(self.groupby("file", observed=True))
        TRUTH = self.elsa.truth.combos
        # precompute
        _ = (
            self.normw, self.normn, self.norme, self.norms, self.iou, self.ibox,
            TRUTH.normw, TRUTH.normn, TRUTH.norme, TRUTH.norms, TRUTH.ibox
        )

        groupby = (
            self
            # cast to dataframe to avoid magicpandas behavior
            .pipe(pd.DataFrame)
            .groupby("file", observed=True)
        )
        it = tqdm(groupby, total=size)
        bounds = 'normw norme norms normn'.split()
        for i, row in it:
            boxes = torch.as_tensor(row[bounds].values)
            scores = torch.tensor(row.iou.values)
            labels = torch.tensor(row.ibox.values)
            preds.append(dict(
                boxes=boxes,
                labels=labels,
                scores=scores,
            ))

            truth = TRUTH.loc[row.ibox]
            boxes = torch.as_tensor(truth[bounds].values)
            labels = torch.tensor(truth.ibox.values)
            target.append(dict(
                boxes=boxes,
                labels=labels,
            ))

        metric.update(preds, target)
        result = metric.compute()
        return result

    @cached_property
    def ap_torch(self):
        iou_thresholds = [0.75, 0.8, 0.85, 0.9]
        metric = MeanAveragePrecision(
            iou_type="bbox",
            iou_thresholds=iou_thresholds,
            max_detection_thresholds=[100, 1000, 1000]
        )
        target = []
        preds = []
        size = len(self.groupby("file", observed=True))
        TRUTH = self.elsa.truth.combos
        # precompute
        _ = (
            self.normw, self.normn, self.norme, self.norms, self.iou, self.ibox,
            TRUTH.normw, TRUTH.normn, TRUTH.norme, TRUTH.norms, TRUTH.ibox,
            self.input, self.target,
        )

        groupby = (
            self
            # cast to dataframe to avoid magicpandas behavior
            .pipe(pd.DataFrame)
            .groupby("file", observed=True)
        )
        it = tqdm(groupby, total=size)
        bounds = 'normw norme norms normn'.split()
        for i, row in it:
            boxes = torch.as_tensor(row[bounds].values)
            scores = torch.tensor(row.iou.values)
            labels = torch.tensor(row.input.values)
            preds.append(dict(
                boxes=boxes,
                labels=labels,
                scores=scores,
            ))

            truth = TRUTH.loc[row.ibox]
            boxes = torch.as_tensor(truth[bounds].values)
            labels = torch.tensor(row.target.values)
            target.append(dict(
                boxes=boxes,
                labels=labels,
            ))

        metric.update(preds, target)
        result = metric.compute()
        return result

    @magic.column
    def cdba(self) -> magic[bool]:
        """Confidence-based Dynamic Box Aggregation"""
        result = np.ones_like(self.index, dtype=bool)
        loc = self.score_range >= .2
        loc &= self.score <= self.score_max - .2
        result &= ~loc

        # subset the groups by maximal scores, and filter invalid groups
        loc = Series(False, index=self.index)
        loc |= self.loc[result].is_invalid
        result &= ~loc

        # filter where iou < .85
        # todo: we have to decide between 85 and 90
        loc = self.iou >= .9
        result &= loc

        return result

    @cached_property
    def filtered(self) -> Self:
        # todo: when I implement this as magic cached, self.sirus is dropped
        """return a subset of the grid which passed the filter"""
        loc = self.cdba
        result = self.loc[loc].pipe(self)
        return result

    @magic.column
    def score_range(self):
        """range of the scores at the same ibox"""
        max = (
            self
            .groupby('ibox')
            .score
            .max()
        )
        min = (
            self
            .groupby('ibox')
            .score
            .min()
        )
        result = (
            (max - min)
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def score_max(self):
        """max of the scores at the same ibox"""
        result = (
            self
            .groupby('ibox')
            .score
            .max()
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def lse_log_n(self):
        """log sum exp of the scores at the same ibox"""

        def apply(scores: Series) -> float:
            return np.log(np.sum(np.exp(scores)))

        result = (
            self
            .groupby('ibox')
            .score
            .apply(apply)
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def iou(self):
        """Intersection over Union"""
        intersection = util.intersection(self.truth, self)
        union = util.union(self.truth, self)
        result = intersection / union
        return result

    @magic.column
    def imax(self) -> magic[float]:
        """The index of the maximum score for the same ibox"""
        result = (
            self.score
            .groupby(self.ibox)
            .idxmax()
            .loc[self.ibox]
            .values
        )
        return result

    @magic.column
    def max_score(self) -> magic[float]:
        """The maximum score for the same ibox"""
        return self.score.loc[self.imax].values

    @magic.cached.property
    def elsa(self) -> Elsa:
        ...

    def scored(
            self,
            score=None,
            *args,
            **kwargs,
    ) -> elsa.evaluation.scored.Scored:
        msg = 'Cannot rescore an already scored grid'
        raise NotImplementedError(msg)

    @magic.column
    def mcauprc(self) -> Series[float]:
        import torch
        scored = self.__outer__
        input = torch.tensor(scored.input.values)
        target = torch.tensor(scored.target)
        from torcheval.metrics import MulticlassAUPRC
        metric = MulticlassAUPRC(num_classes=input.size(1))
        metric.update(input, target)
        result = (
            metric
            .compute()
            .cpu()
            .numpy()
        )
        return result
