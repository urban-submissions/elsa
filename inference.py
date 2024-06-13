from __future__ import annotations
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from pathlib import Path
from typing import *
from elsa.predictv3.predict import PromptIteration
from elsa.root import Elsa

if __name__ == '__main__':

    bing = '/home/marco.cipriano/data/label_1k/bing/images'
    google = '/home/marco.cipriano/data/label_1k/google/images'
    unified = bing, google

    labels = Path(
        "/home/marco.cipriano/projects/elsa/gt_data/triple_inspected_May23rd/merged/label_id_dict_after_distr_thresholding.csv"
    )
    elsa: Elsa = Elsa.from_unified(files=unified, labels=labels)

    iterations: Iterator[PromptIteration] = elsa.predict.gdino3p(outdir="/home/marco.cipriano/results_rescale")
    with ThreadPoolExecutor() as threads:
        futures = []
        submit = pd.DataFrame.to_parquet
        for iteration in iterations:
            frame = iteration.logits
            submit = iteration.save_logits
            file = iteration.outpath
            future = threads.submit(submit, frame, file)
            future.result()
            # futures.append(future)

        for future in futures:
            pass