from __future__ import annotations

import warnings

from elsa import Elsa
warnings.warn('Loading a module personalized for dhodcz2')
truth = '/home/arstneio/Downloads/modified/bing/yolo/labels'
files = '/home/arstneio/Downloads/modified/bing/yolo/images'
labels = '/home/arstneio/Downloads/modified/bing/yolo/notes.json'
images = '/home/arstneio/Downloads/modified/bing/coco/result.json'

elsa = Elsa.from_bing(
    truth=truth,
    files=files,
    labels=labels,
    images=images,
)
label = (
    f"alone; group; sitting; standing; walking; running; biking; "
    f"mobility aids; riding carriage; public service/cleaning; "
    f"talking on phone; load/unload packages from car/truck; "
    f"phone interaction; street vendors; talking; pet interactions; "
    f"dining; pushing stroller or shopping cart; crossing crosswalk; "
    f"person; an individual"
).split('; ')


# select where combo is entirely in those labels, and is at least CS
rephrase = elsa.truth.unique.rephrase
loc = (
    rephrase.label
    .isin(label)
    .groupby(level='irephrase')
    .all()
    .loc[rephrase.irephrase]
    .values
)
irephrase = rephrase.irephrase[loc]
prompts = elsa.prompts.irephrase.isin(irephrase)
loc = elsa.truth.unique.includes(meta='condition')
loc &= elsa.truth.unique.includes(meta='state')
isyns = elsa.truth.unique.isyns[loc]
prompts &= elsa.prompts.isyns.isin(isyns)

# todo: replace .c with combos.is_invalid to avoid all invalids
# exclude invalid combos
loc = elsa.truth.combos.invalid.c
isyns = elsa.truth.combos.isyns.loc[loc]
prompts &= ~elsa.prompts.isyns.isin(isyns)
invalid_files = elsa.truth.files.file.loc[loc].unique()

# select files that contain at least one of the prompts
isyns = elsa.prompts.isyns.loc[prompts].unique()
truth = elsa.truth.combos
loc = truth.isyns.isin(isyns)
# exclude invalid combos
loc &= ~truth.invalid.c
# exclude files where any combo is invalid
filenames = (
    truth.file
    .loc[loc]
    .unique()
)
files = truth.files.file.isin(filenames)
files &= ~truth.files.file.isin(invalid_files)
# select first 30 files
# files &= files.cumsum() <= 30


"""
iterate across prediction;
    iterate across files

/image
    image.png
    /prompt
        label1.png
        label2.png
        label3.png
"""

