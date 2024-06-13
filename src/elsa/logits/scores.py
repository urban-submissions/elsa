from __future__ import annotations

import math
from functools import *
from scipy.special import logsumexp
from typing import *

import magicpandas as magic

if False:
    from .logits import Logits


class Score(magic.column):
    def __set_name__(self, owner: Selection, name):
        super().__set_name__(owner, name)
        owner.scoring.add(name)


class Selection(magic.Magic):
    __outer__: Scores
    scoring = set()

    @property
    def logits(self):
        return self.__outer__.__outer__

    @Score
    def lse(self) -> magic[float]:
        logits = self.logits
        result = logsumexp(logits.confidence, axis=1)
        return result

    @Score
    def loglse(self) -> magic[float]:
        logits = self.logits
        result = logsumexp(logits.confidence, axis=1)
        result -= math.log(self.n)
        return result

    @Score
    def avglse(self) -> magic[float]:
        logits = self.logits
        result = logsumexp(logits.confidence * self.n, axis=1)
        result /= self.n
        return result

    @property
    def n(self) -> int:
        return len(self.logits.confidence.columns)

    @Score
    def argmax(self):
        """max out of the columns for the row"""
        logits = self.logits
        result = logits.confidence.max(axis=1)
        return result


class Selected(Selection):

    @property
    def logits(self):
        return super().logits.without_extraneous_spans


class Scores(magic.Frame):
    __outer__: Logits
    whole = Selection()
    selected = Selected()

    @property
    def everything(self):
        # todo: this is terribly slow for what it is
        whole = self.whole
        selected = self.selected
        keys = whole.scoring
        for key in keys:
            getattr(whole, key)
        for key in keys:
            getattr(selected, key)
        return self



    def __from_inner__(self) -> Self:
        index = self.__outer__.index
        result = self(index=index)
        return result

