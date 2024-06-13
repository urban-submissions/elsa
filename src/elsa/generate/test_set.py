from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pandas import Series
from pathlib import Path

from elsa import Logits
from elsa.root import Elsa

elsa = Elsa.from_unified()
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 2 c, 4 cs, 3 csa, 1 csaoV}

indir = '/home/arstneio/Downloads/predictions'
indir = Path(indir)
inpaths = Series(indir.rglob('*.parquet'))
inprompts = Series([path.stem for path in inpaths])

_ = elsa.prompts.isyns, elsa.prompts.level
prompts = elsa.prompts.drop_duplicates('isyns')
loc = elsa.prompts.natural.isin(inprompts)
prompts = prompts.loc[loc]

loc = prompts.level == 'c'
isyns = prompts.isyns.loc[loc].iloc[:2]
c = elsa.prompts.isyns.isin(isyns)

loc = prompts.level == 'cs'
isyns = prompts.isyns.loc[loc].iloc[:4]
cs = elsa.prompts.isyns.isin(isyns)

loc = prompts.level == 'csa'
isyns = prompts.isyns.loc[loc].iloc[:3]
csa = elsa.prompts.isyns.isin(isyns)

loc = prompts.level == 'csao'
isyns = prompts.isyns.loc[loc].iloc[:1]
csao = elsa.prompts.isyns.isin(isyns)

prompts = c | cs | csa | csao
isyns = elsa.prompts.isyns.loc[prompts]

combos = elsa.truth.combos
combos.file
file = (
    combos.file
    .groupby(combos.isyns.values, sort=False)
    .first()
    .loc[isyns]
    .unique()
)

loc = combos.isyns.isin(isyns)
loc &= ~combos.file.isin(file)
loc &= np.cumsum(loc) <= (10 - len(file))
loc |= combos.file.isin(file)
files = (
    combos.file
    .loc[loc]
    .unique()
)
assert len(files) == 10

nunique = (
    elsa.prompts
    .loc[prompts]
    .groupby('level')
    .isyns
    .nunique()
)
assert nunique.sum() == 10
assert nunique.sum() == 10

loc = inprompts.isin(elsa.prompts.natural.loc[prompts])
inpaths = inpaths.loc[loc]
inprompts = inprompts.loc[loc]

list_isyns = (
    elsa.prompts.isyns
    .set_axis(elsa.prompts.natural)
    .loc[inprompts]
)
assert list_isyns.nunique() == 10

outdir = Path('/home/arstneio/Downloads/scores')
force = True

with ThreadPoolExecutor() as threads:
    futures = []
    total = len(inpaths) * len(files)
    i = 0
    # for inpath in inpaths:
    # for inpath, isyns in zip(inpaths, list_isyns):
    it = zip(inpaths, list_isyns, inprompts)
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
