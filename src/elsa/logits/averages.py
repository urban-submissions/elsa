from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future

import pandas as pd
from functools import *
from pathlib import Path
from typing import *

import magicpandas as magic

if False:
    from elsa.root import Elsa


class Averages(magic.Frame):
    @classmethod
    def from_directory(
            cls,
            indir,
            elsa: Elsa = None
    ) -> Self:
        """Determine averages from a directory of logit scores"""
        indir = Path(indir)
        inpaths = list(indir.rglob('*.parquet'))

        with ThreadPoolExecutor() as threads:
            from_file = pd.read_parquet

            def it_scores() -> Iterator[pd.DataFrame]:
                """preemptively loads the scores from the files"""
                it: Iterator[Future] = (
                    threads.submit(from_file, infile)
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

            concat = []
            scores = it_scores()
            scores = list(scores)
            keys = 'prompt file'.split()
            for score in scores:
                score = score.set_index(keys)
                loc = [
                    'scores.'
                    in column
                    for column in score.columns
                ]
                means = (
                    score
                    .loc[:, loc]
                    .groupby(level=keys)
                    .mean()
                )
                concat.append(means)

            result = (
                pd.concat(concat)
                .pipe(cls)
            )
            # if elsa is not None:
            #     prompt = averages.prompt.unique()
            #     loc = ~prompt.unique().isin(elsa.prompts.natural)
            #     # if loc.any():
            #     #     x = loc.sum()
            result.elsa = elsa
            return result

    @magic.cached.property
    def elsa(self) -> Elsa:
        raise ValueError('elsa must be set')

    @magic.index
    def prompt(self) -> magic[str]:
        """The prompt that the logits were averaged from"""

    @magic.index
    def file(self):
        """The file that the logits were averaged from"""

    @magic.column
    def level(self) -> magic[str]:
        """the level of the prompt eg. c, cs, csa."""
        prompts = self.elsa.truth.unique.rephrase.prompts
        _ = prompts.level
        result = (
            prompts
            .drop_duplicates('natural')
            .set_index('natural')
            .loc[self.prompt]
            .level
            .values
        )
        return result

    @cached_property
    def scores(self) -> Self:
        """only the columns that contain scores."""
        loc = [
            'scores.'
            in column
            for column in self.columns
        ]
        result = self.loc[:, loc]
        return result

    @cached_property
    def average_per_prompt(self) -> Self:
        """the average score per prompt."""
        result = (
            self.scores
            .groupby(level='prompt')
            .mean()
        )
        return result


    @cached_property
    def average_per_level(self) -> Self:
        """the average score per level."""
        _ = self.level
        result = (
            self.scores
            .groupby(self.level)
            .mean()
        )
        return result


if __name__ == '__main__':

    averages = Averages.from_directory(
        '/tmp/scores',
        elsa=elsa
    )
    averages.average_per_level
    averages.average_per_prompt

    averages.head()
