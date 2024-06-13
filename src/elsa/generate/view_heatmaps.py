from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import os
from PIL.Image import Image
from matplotlib.figure import Figure
from pathlib import Path
from typing import *

from elsa import util
from elsa.generate.premade import elsa, files, prompts
from elsa.predictv3.predict import ImageIteration
from elsa.predictv3.predict import dhodcz2

# select prompts that contain 3 unique labels
truth = elsa.truth
combos = truth.combos
loc = combos.get_nunique_labels() == 3
isyns = combos.isyns.loc[loc]
prompts &= elsa.prompts.isyns.isin(isyns)

iterations: Iterator[ImageIteration] = elsa.predict.gdino3p(
    files=files,
    prompts=prompts,
    checkpoint_path=dhodcz2,
    yield_image=True,
    yield_prompt=False,
)

if __name__ == '__main__':
    outdir = Path('~/Downloads/heatmaps').expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    def submit_image(p: Path, image: Image):
        image.save(p)
        image.close()

    def submit_figure(p: Path, figure: Figure):
        figure.savefig(p)
        plt.close(figure)

    with ThreadPoolExecutor() as threads:
        futures = []
        for iteration in iterations:
            # create directory
            file = util.trim_path(iteration.file)
            labels = ', '.join(iteration.selection.labels)
            dir = Path(outdir, file)
            dir.mkdir(parents=True, exist_ok=True)

            # plot overlaid heatmaps
            overlaid: Image = iteration.selection.view_overlaid_heatmaps()
            p = dir / f'{labels} overlaid.png'
            future = threads.submit(submit_image, p, overlaid)
            futures.append(future)

            # plot stacked heatmaps
            stacked: Figure = iteration.selection.view_stacked_heatmaps()
            p = dir / f'{labels} stacked.png'
            future = threads.submit(submit_figure, p, stacked)
            futures.append(future)

"""
/img
    /prompt
        original.png
        overlaid.png
        stacked.png
"""
