# ELSA: evaluating localization of social activities

## Dataset

Inside the `dataset` folder we include two subfolders `(google | bing)` containing csv files named `matched_rows.csv`.
The `csv` file contains coordinates, ids, datetime and filenames of the images included in ELSA.
There are several libraries available which allows to download from the Google and Bing APIs.
For your convenience we included two scripts which use the streetlevel library and require no API Key:

```
pip install streetlevel
cd dataset/bing
python download_images.py
```

## Benchmark

### Installation

Install mamba and create a virtual environment with python 3.12

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh -b -p ~/mambaforge
~/mambaforge/bin/mamba init
source ~/.bashrc
mamba create -n elsa python=3.12.3
mamba activate elsa

```

Install Grounding DINO (official)

```bash
cd ~
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```


Clone the ELSA repository and install
```bash
cd ~
git clone git@github.com:urban-submissions/elsa.git
cd SIRiUS
pip install -e .
```

Install Open-GroundingDino in models after setting up CUDA env variables

```bash
cd models
git clone https://github.com/longzw1997/Open-GroundingDino.git
cd Open-GroundingDino/
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

cd models/GroundingDINO/ops
python setup.py build install
python test.py
```

Update PYTHONPATH to include Open-GroundingDino
```bash
export PYTHONPATH=/home/yourusername/projects/elsa:/home/yourusername/projects/elsa/models/Open-GroundingDino
```

Upgrade the supervision module
```bash
pip install --upgrade supervision
```


Download weights
```bash
cd /home/yourusername/projects/elsa/weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
Download the configuration file `cfg_odvg.py` from the online repo

### Generate predictions

You can generate the predictions from the main project folder:
```python
from elsa import *

bing = 'yourpath/label_1k/bing/images'
google = 'yourpath/label_1k/google/images'
files = bing, google
elsa = Elsa.from_unified(files)

predictions = 'yourpath/to/predictions'
config = '/yourpath/to/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py'
checkpoint='/yourpath/to/gdinot-coco-ft.pth'

elsa.predict(
    outdir=predictions,
    batch_size=16,
    config=config,
    checkpoint=checkpoint,
)
```

### Evaluate
Pass the previous predictions folder to be concatenated into a single file
based on the scores which are above a threshold. Later this file can be re-used
to generate the evaluation. You can then select which of the scores to use for
evaluation and access the f1 metric.
```python

from elsa import *
bing = 'yourpath/label_1k/bing/images'
google = 'yourpath/label_1k/google/images'
files = bing, google
elsa = Elsa.from_unified(files)

predictions = 'yourpath/to/predictions'
# if concat does not exist, the predictions above a threshold will
# be concatenated and saved to this file.
concat = 'yourpath/to/concat.parquet'
evaluate = elsa.evaluate(
    concatenated=concat,
    logits=predictions,
)
scored = evaluate.scored('selected.loglse')
scored.f1
```
