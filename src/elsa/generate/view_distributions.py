from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import matplotlib.figure
import os
from pathlib import Path
from typing import *

from elsa import Elsa
from elsa import util
from elsa.predictv3.predict import ImageIteration
from elsa.predictv3.predict import dhodcz2
from elsa.generate.premade import elsa, files, prompts

iterations: Iterator[ImageIteration] = elsa.predict.gdino3p(
    files=files,
    prompts=prompts,
    checkpoint_path=dhodcz2,
    yield_image=True,
    yield_prompt=False,
)

if __name__ == '__main__':
    outdir = Path('~/Downloads/distributions').expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor() as threads:
        futures = []
        for iteration in iterations:
            file = util.trim_path(iteration.file)
            path = Path(outdir, file)
            path.mkdir(parents=True, exist_ok=True)
            distributions = iteration.selection.view_distributions()
            for label, distribution in zip(iteration.selection.labels, distributions):
                distribution: matplotlib.figure.Figure
                prompt = iteration.prompt.replace('.', '')
                p = Path(path, f"'{label}' from {prompt}'.png")
                threads.submit(distribution.savefig, str(p))

        for future in futures:
            future.result()

"""
/image
    image.pn
    /prompt
        label1.png
        label2.png
        label3.png
"""
