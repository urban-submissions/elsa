from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import shutil
from pandas import Series
from pathlib import Path

from elsa import Logits
from elsa.generate.test_set import (
    files as FILES,
    prompts as PROMPTS,
    elsa as elsa
)

files = elsa.files.file.loc[FILES.values]
allowed_prompts = elsa.prompts.natural.loc[PROMPTS.values]

indir = '/home/arstneio/Downloads/predictions'
indir = Path(indir)
inpaths = Series(indir.rglob('*.parquet'))
prompts = Series([path.stem for path in inpaths])
loc = prompts.isin(elsa.prompts.natural)
inpaths = inpaths.loc[loc]
prompts = prompts.loc[loc]
loc =  prompts .isin(allowed_prompts)
prompts = prompts.loc[loc]

# assert allowed_prompts.isin(elsa.prompts.natural).all()

loc = prompts.isin(allowed_prompts)
prompts = prompts.loc[loc]
inpaths = inpaths.loc[loc]

list_isyns = (
    elsa.prompts.isyns
    .set_axis(elsa.prompts.natural)
    .loc[prompts]
)
outdir = Path('/home/arstneio/Downloads/scores')
force = False
# shutil.rmtree(outdir, ignore_errors=True)

force = True
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# todo: if prompt is actually in the file, should be file/CARDINAL/prompt.png
elsa.truth.combos

truth = elsa.truth.combos
haystack = set(zip(truth.file, truth.isyns))

loc = truth.file == '103331020022110310_x4_cropped'
truth.loc[loc]

with ThreadPoolExecutor() as threads:
    futures = []
    total = len(inpaths) * len(files)
    i = 0
    # for inpath in inpaths:
    # for inpath, isyns in zip(inpaths, list_isyns):
    it = zip(inpaths, list_isyns, prompts)
    for inpath, isyns, prompt in it:
        logits = Logits.from_file(inpath, elsa=elsa)
        # todo: if isyns is in truth

        for file in files:
            print(f'{i}/{total}')

            i += 1
            p = inpath.relative_to(indir)
            parts = p.parts
            # # if (file, isyns) in haystack:
            #     parts = (parts[0].upper(),) + parts[1:]

            new_path = Path(*parts)
            outpath = (
                    outdir
                    / file
                    / new_path
                    .with_suffix('.png')
            )
            if (
                    not force
                    and outpath.exists()
            ):
                continue

            outpath.parent.mkdir(parents=True, exist_ok=True)
            image = logits.view_top(file=file)
            future = threads.submit(image.save, outpath)
            futures.append(future)

    for future in as_completed(futures):
        future.result()

"""
outdir/
    file/
        cardinal/
            prompt.png
"""
