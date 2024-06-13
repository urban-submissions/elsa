from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import os
from pathlib import Path
from typing import *

from elsa import util
from elsa.generate.premade import elsa, files, prompts
from elsa.predictv3.predict import ImageIteration
from elsa.predictv3.predict import dhodcz2

iterations: Iterator[ImageIteration] = elsa.predict.gdino3p(
    files=files,
    prompts=prompts,
    checkpoint_path=dhodcz2,
    yield_image=True,
    yield_prompt=False,
)

if __name__ == '__main__':
    outdir = Path('~/Downloads/stacked_heatmaps').expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor() as threads:
        futures = []
        for iteration in iterations:
            iteration.selection.norm
            file = util.trim_path(iteration.file)
            path = Path(outdir, file)
            path.mkdir(parents=True, exist_ok=True)
            heatmaps = iteration.selection.view_stacked_heatmaps()
            p = f"{' '.join(iteration.selection.labels)}.png"
            future = threads.submit(heatmaps.savefig, str(Path(path, f"{iteration.prompt}.png")))
            futures.append(future)

        for future in futures:
            future.result()

"""
/img
    /prompt
        stacked.png
"""