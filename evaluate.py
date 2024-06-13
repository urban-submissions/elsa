
from elsa import Elsa
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.WARNING)

bing = '/home/marco.cipriano/data/label_1k/bing/images'
google = '/home/marco.cipriano/data/label_1k/google/images'
unified = bing, google

labels = Path(
    "/home/marco.cipriano/projects/elsa/gt_data/triple_inspected_May23rd/merged/label_id_dict_after_distr_thresholding.csv"
)
elsa: Elsa = Elsa.from_unified(files=unified, labels=labels)

evaluation = elsa.evaluate(
    # if checkpoint does not exists, reads from logits dir and concats into checkpoint
    concatenated='/home/marco.cipriano/res_parquets/selected.loglse_.3.parquet',
    logits='/home/marco.cipriano/results',
)


## Our re-ranking
scored = evaluation.scored('selected.loglse')
result = scored.average_precision
df = pd.DataFrame(result)
# ensuring name is the first column for clarity and dumping
columns = ['level'] + [col for col in df.columns if col != 'level']
df = df[columns]
df.to_csv("/home/marco.cipriano/res_parquets/map_loglse.csv")

## Max
scored = evaluation.scored('whole.argmax')
result = scored.average_precision
df = pd.DataFrame(result)
# ensuring name is the first column for clarity and dumping
columns = ['level'] + [col for col in df.columns if col != 'level']
df = df[columns]
df.to_csv("/home/marco.cipriano/res_parquets/map_argmax.csv")