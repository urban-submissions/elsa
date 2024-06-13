from __future__ import annotations

import warnings

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from typing import *

from elsa.files import unified_dhodcz
from elsa.predictv3.predict import PromptIteration
from elsa.root import Elsa

elsa = Elsa.from_unified(files=unified_dhodcz)
truth = elsa.truth
combos = truth.combos
prompts = np.full(elsa.prompts.shape[0], True)

# must include state
loc = combos.includes(meta='state')
isyns = combos.isyns.loc[loc]
prompts &= elsa.prompts.isyns.isin(isyns)

# must include activity
loc = combos.includes(meta='activity')
isyns = combos.isyns.loc[loc]
prompts &= elsa.prompts.isyns.isin(isyns)

# select relevant files
isyns = elsa.prompts.isyns.loc[prompts]
loc = combos.isyns.isin(isyns)
file = combos.file.loc[loc]

# at least two files with 3 or more boxes
a = elsa.files.file.isin(file)
b = elsa.files.nboxes.values >= 3
b &= a
b &= b.cumsum() <= 2
# at least one file with 5 or more boxes
c = elsa.files.nboxes.values >= 5
c &= a
c &= c.cumsum() <= 1
files = b | c

# files = a.copy()
# files &= a.cumsum() <= 1

# loc = combos.includes('people')
# loc &= combos.includes('standing')
# loc &= combos.includes('crossing crosswalk')
# isyns = combos.isyns.loc[loc]
# prompts &= elsa.prompts.isyns.isin(isyns)
#
files = elsa.files.file == '103330322123201010_x4_cropped'
files
# files = elsa.files.nfile <= 1

if __name__ == '__main__':
    iterations: Iterator[PromptIteration] = (
        elsa.predict.gdino3p.batched_without_interpolation(
        prompts=prompts,
            files=files
        )
    )

    from itertools import chain
    gdino = elsa.predict.gdino3p

    with ThreadPoolExecutor() as threads:
        futures = []
        submit = pd.DataFrame.to_parquet
        for iteration in iterations:
            frame = iteration.logits
            submit = iteration.save_logits
            file = iteration.outpath
            future = threads.submit(submit, frame, file)
            futures.append(future)

            # f = file.parent / f'{iteration.prompt}.png'
            # image = iteration.logits.view_max()
            # future = threads.submit(submit_image, file, image)
            # futures.append(future)

        for future in futures:
            future.result()
