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
    outdir = Path('~/Downloads/stacked_distributions').expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor() as threads:
        futures = []
        for iteration in iterations:
            iteration.selection.norm
            file = util.trim_path(iteration.file)
            path = Path(outdir, file)
            path.mkdir(parents=True, exist_ok=True)
            distributions = iteration.selection.view_stacked_distributions()
            p = f"{' '.join(iteration.selection.labels)}.png"
            future = threads.submit(distributions.savefig, str(Path(path, f"{iteration.prompt}.png")))
            futures.append(future)

        for future in futures:
            future.result()


"""
/image
    image.png
    labels.png
"""
