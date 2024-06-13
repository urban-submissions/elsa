from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import pandas as pd
from pandas import Series
from pathlib import Path
from typing import *

from elsa.predictv3.predict import PromptIteration

if False:
    from elsa.root import Elsa


def predict(
        elsa: Elsa,
        outdir: Path | str,
        batch_size: int,
        checkpoint: Path | str,
        config: Path | str,
        files: Series[bool] = None,
        prompts: Series[bool] = None,
        force: bool = False,
        method: str = 'batched_without_interpolation',
):
    """
    elsa:
        the Elsa instance that encapsulates the image metadata,
        ground truth annotations, prompts, and other resources
    outdir:
        The output direction to which the logits are to be saved
    checkpoint:
        the path to the GroundingDino checkpoint file, e.g.
        path/to/gdino-coco-ft.pth
    config:
        the path to the GroundingDino configuration file, e.g.
        path/to/GroundingDINO_SwinT_OGC.py
    batch_size:
        the maximum amount of images to be processed in a single batch
    files:
        a boolean mask aligned with Elsa.files that specifies
        for which files inference is to be run
    prompts:
        a boolean mask aligned with Elsa.prompts that specifies
        for which prompts inference is to be run
    force:
        True:
            overwrite the logits if they already exist
    """

    if not (
            files is None
            or isinstance(files, (list, Series, np.ndarray))
    ):
        msg = (
            f'files must be a list, Series, or numpy array, '
            f'but got {type(files)}. This parameter is for boolean '
            f'masking the files, not for specifying the files themselves. '
            f'The files themselves must be encapsulated in a Elsa object.'
        )
        raise ValueError(msg)
    if not (
            prompts is None
            or isinstance(prompts, (list, Series, np.ndarray))
    ):
        msg = (
            f'prompts must be a list, Series, or numpy array, '
            f'but got {type(prompts)}. This parameter is for boolean '
            f'masking the prompts, not for specifying the prompts themselves. '
            f'The prompts themselves must be encapsulated in a Elsa object.'
        )
        raise ValueError(msg)

    outdir = (
        Path(outdir)
        .expanduser()
        .resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)
    # func = getattr(elsa.predict.gdino3p, method)
    func = getattr(elsa.gdino, method)
    it: Iterator[PromptIteration] = func(
        config=config,
        checkpoint=checkpoint,
        outdir=outdir,
        files=files,
        prompts=prompts,
        force=force,
        batch_size=batch_size,
    )

    serialize = pd.DataFrame.to_parquet

    def submit(result, outpath: Path, *args, **kwargs):
        nonlocal serialize
        result.attrs = {}
        outpath.parent.mkdir(parents=True, exist_ok=True)
        serialize(result, outpath, *args, **kwargs)
        return outpath

    with ThreadPoolExecutor() as threads:
        # noinspection PyTypeChecker

        futures = [
            threads.submit(
                submit,
                # outdir / f'{prompt.outpath}.{extension}',
                prompt.logits,
                prompt.outpath,
            )
            for prompt in it
        ]
    for future in as_completed(futures):
        future.result()


if __name__ == '__main__':
    from elsa.generate.three_per_condition import elsa, files, prompts

    predict(elsa, files, prompts)
