from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL.Image import Image
from pathlib import Path
from typing import *

# from elsa.generate.premade import elsa, files, prompts
from elsa.generate.three_per_condition import elsa, files, prompts
from elsa.predictv3.predict import PromptIteration



def submit_image(p: Path, image: Image):
    image.save(p)
    image.close()


truth = elsa.truth
combos = truth.combos
loc = combos.includes('individual')
loc &= combos.includes('on phone')
loc &= combos.includes('standing up')
loc &= combos.includes('crossing road')
prompts = elsa.prompts.isyns.isin(combos.isyns.loc[loc])

isyns = elsa.prompts.isyns.loc[prompts]
loc = combos.isyns.isin(isyns)
file = elsa.truth.combos.file.loc[loc]
files = elsa.truth.files.file.isin(file)
files &= files.cumsum() <= 20

if __name__ == '__main__':
    iterations: Iterator[PromptIteration] = (
        elsa.predict.gdino3p(
            prompts=prompts
        )
    )
    with ThreadPoolExecutor() as threads:
        futures = []
        submit = pd.DataFrame.to_parquet
        for iteration in iterations:
            frame = iteration.logits
            submit = iteration.save_logits
            file = iteration.outpath
            future = threads.submit(submit, frame, file)
            futures.append(future)


        for future in futures:
            future.result()

# f = file.parent / f'{iteration.prompt}.png'
# image = iteration.logits.view_max()
# future = threads.submit(submit_image, file, image)
# futures.append(future)

print('done')

"""
/maxes
    prompt.parquet
    prompt.png
"""
