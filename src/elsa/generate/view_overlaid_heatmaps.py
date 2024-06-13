from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os
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
    outdir = Path('~/Downloads/heatmap_overlays').expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor() as threads:
        futures = []
        for iteration in iterations:
            file = util.trim_path(iteration.file)
            path = Path(outdir, file)
            path.mkdir(parents=True, exist_ok=True)
            heatmap = iteration.selection.view_overlaid_heatmaps()

            labels = ', '.join(iteration.selection.labels)
            p = f"{labels}.png"
            future = threads.submit(heatmap.savefig, str(Path(path, f"{iteration.prompt}.png")))
            futures.append(future)

"""
/img
    /prompt
        overlaid.png
"""
