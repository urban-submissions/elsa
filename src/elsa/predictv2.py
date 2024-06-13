from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future

import logging
import numpy as np
import os
import pandas as pd
import tempfile
import warnings
# from autodistill.detection import CaptionOntology
from functools import cached_property
from numpy import ndarray
from pandas import DataFrame
from pandas import Series
from pathlib import Path
from typing import *

import magicpandas as magic
# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
# import tools as open_groundingdino
from elsa.predictv3.predict import GDino3P

if False:
    from .root import Elsa
    from autodistill_grounding_dino import GroundingDINO
    from autodistill_owlv2 import OWLv2
CaptionOntology = {}
"""
todo: return to this if there is time to improve the module; 
there are speed gains to be made from loading the models in parallel 
however it randomly (not everytime) throws the following error:

ModuleNotFoundError: No module named 'tmpboztqqad'
ModuleNotFoundError: No module named 'tmp6x3bny7s'

File ~/PycharmProjects/elsa/venv/lib/python3.12/site-packages/groundingdino/util/slconfig.py:185, in SLConfig.fromfile(filename)
    183 @staticmethod
    184 def fromfile(filename):
--> 185     cfg_dict, cfg_text = SLConfig._file2dict(filename)
    186     return SLConfig(cfg_dict, cfg_text=cfg_text, filename=filename)

File ~/PycharmProjects/elsa/venv/lib/python3.12/site-packages/groundingdino/util/slconfig.py:90, in SLConfig._file2dict(filename)
     88 sys.path.insert(0, temp_config_dir)
     89 SLConfig._validate_py_syntax(filename)
---> 90 mod = import_module(temp_module_name)
     91 sys.path.pop(0)
     92 cfg_dict = {
     93     name: value for name, value in mod.__dict__.items() if not name.startswith("__")
     94 }

File /usr/lib/python3.12/importlib/__init__.py:90, in import_module(name, package)
     88             break
     89         level += 1
---> 90 return _bootstrap._gcd_import(name[level:], package, level)
"""

np.set_printoptions(suppress=True)


# def batch(
#         prompt: str,
#         truth: Truth,
#         model: type[GroundingDINO | OWLv2],
#         isyn: ndarray,
#         ilabel: ndarray,
# ) -> Predictions:
#     confidence = []
#     wsen = []
#     ontology = CaptionOntology({prompt: prompt})
#     model = model(ontology=ontology)
#     for path in truth.path.values:
#         detections = model.predict(path)
#         wsen.append(detections.xyxy)
#         confidence.append(detections.confidence)
#
#     isyn = isyn
#     # what is the length in labels of the prompt
#     isyns = len(isyn)
#     # how many predictions made for each file iterated
#     predictions_per_file = np.fromiter(map(len, wsen), int, len(wsen))
#     # broadcast the detections
#     wsen = np.concatenate(wsen).repeat(isyns)
#     # broadcast the confidence
#     confidence = np.concatenate(confidence).repeat(isyns)
#     # broadcast the ifile
#     ifile = truth.ifile.repeat(predictions_per_file).repeat(isyns)
#     # broadcast the isyn, tiled to get 0 1 2 0 1 2 etc
#     isyn = np.tile(isyn, predictions_per_file.sum())
#     # broadcast the ilabel
#     ilabel = np.tile(ilabel, predictions_per_file.sum())
#
#     result = Predictions(dict(
#         w=wsen[:, 0],
#         s=wsen[:, 1],
#         e=wsen[:, 2],
#         n=wsen[:, 3],
#         ifile=ifile,
#         confidence=confidence,
#         isyn=isyn,
#         ilabel=ilabel,
#     ))
#
#     # serialize(result, outpath, *args, **kwargs)
#     # todo: return path or dataframe?
#     return result

# todo: support selecting files
class AutoDistill(
    magic.Magic,
):
    model: type[GroundingDINO | OWLv2]
    __outer__: Predict

    def __call__(
            self,
            outdir: str | Path = None,
            extension: str = 'parquet',
            files=None,
            prompts=None,
            quiet=False,
            force=False,
            *args,
            **kwargs,
            # ) -> Iterator[Path]:
    ) -> Iterator[Future]:

        """
        Parameters
        ----------
        outdir
            destination directory for the predictions
        extension
            file extension to save the predictions
        files: ndarray[bool]
            a boolean mask for which files are to be used
        prompts: ndarray[bool]
            a boolean mask for which prompts are to be used
        quiet
            whether to suppress logging
        force
            whether to overwrite existing files
        args
            to be passed to the serialize function
        kwargs
            to be passed to the serialize function

        Returns
        -------
        Iterator[Future]
        """
        if files is None:
            files = slice(None)
        if prompts is None:
            prompts = slice(None)
        if outdir is None:
            outdir = Path(tempfile.mkdtemp())
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        columns = ('w s e n ifile file confidence isyn label ilabel '
                   'path image_width image_height').split()
        empty = DataFrame(columns=columns)

        if quiet:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            logging.getLogger('transformers').setLevel(logging.ERROR)
            logging.getLogger('PIL').setLevel(logging.ERROR)

        elsa = self.__outer__.__outer__
        truth = elsa.truth
        PROMPTS = truth.unique.rephrase.prompts
        FILES = elsa.files
        _ = PROMPTS.isyns, PROMPTS.natural, elsa.labels.isyn, truth.path, FILES.nfile, FILES.path, FILES.ifile
        prompts = PROMPTS.loc[prompts]

        files = FILES.loc[files]

        message = (
            f'Using {len(prompts)} prompts to predict {len(files)} images '
            f'with {self.model.__class__.__name__}...'
        )
        MODEL = self.model
        self.__logger__.info(message)
        unique = elsa.truth.unique

        ILABEL = (
            elsa.labels
            .reset_index()
            .drop_duplicates('isyn')
            .set_index('isyn')
            .label
        )
        try:
            serialize = getattr(pd.DataFrame, f'to_{extension}')
        except AttributeError:
            raise ValueError(f'Unsupported extension: {extension}')

        def submit(result, outpath, *args, **kwargs):
            nonlocal serialize
            serialize(result, outpath, *args, **kwargs)
            return outpath

        with ThreadPoolExecutor() as threads:
            futures = []
            it = (
                prompts
                .reset_index()
                .itertuples(index=False)
            )
            for prompt in it:
                natural = prompt.natural
                outpath = Path(
                    outdir,
                    f'{natural}.{extension}',
                ).resolve()
                if (
                        not force
                        and outpath.exists()
                ):
                    continue
                isyn = unique.isyn.loc[prompt.isyns]
                label = ILABEL.loc[isyn]

                confidence = []
                wsen = []
                ontology = CaptionOntology({natural: natural})
                model = MODEL(ontology=ontology)
                for path in files.path.values:
                    detections = model.predict(path, )
                    wsen.append(detections.xyxy)
                    confidence.append(detections.confidence)

                if not len(wsen):
                    if (
                            not force
                            and outpath.exists()
                    ):
                        continue
                    future = threads.submit(
                        submit,
                        empty,
                        outpath,
                        *args,
                        **kwargs
                    )
                    futures.append(future)
                    warnings.warn(
                        f'No detections in any file for '
                        f'promt {natural}',
                        UserWarning
                    )
                    yield future
                    continue

                isyn = isyn
                # what is the length in labels of the prompt
                try:
                    isyns = len(isyn)
                except TypeError:
                    # sometimes we get a scalar
                    isyns = 1
                # how many predictions made for each file iterated
                predictions_per_file = np.fromiter(map(len, wsen), int, len(wsen))
                # broadcast
                wsen = np.concatenate(wsen).repeat(isyns, axis=0)
                confidence = np.concatenate(confidence).repeat(isyns)
                file = files.file.values.repeat(predictions_per_file).repeat(isyns)
                ifile = files.ifile.values.repeat(predictions_per_file).repeat(isyns)
                path = files.path.values.repeat(predictions_per_file).repeat(isyns)

                # broadcast the isyn, tiled to get 0 1 2 0 1 2 etc
                isyn = np.tile(isyn, predictions_per_file.sum())
                # broadcast the ilabel
                label = np.tile(label, predictions_per_file.sum())
                image_width = elsa.images.width.loc[ifile].values
                image_height = elsa.images.height.loc[ifile].values
                # todo: apparently sometimes the detections are out of bounds for owl?
                #   for now we're just avoiding owl
                # w = np.clip(wsen[:, 0], 0, image_width)
                # s = np.clip(wsen[:, 1], 0, image_height)
                # e = np.clip(wsen[:, 2], 0, image_width)
                # n = np.clip(wsen[:, 3], 0, image_height)
                w = wsen[:, 0]
                s = wsen[:, 1]
                e = wsen[:, 2]
                n = wsen[:, 3]
                ilabel = elsa.labels.label2ilabel.loc[label].values

                result = DataFrame(dict(
                    w=w,
                    s=s,
                    e=e,
                    n=n,
                    ifile=ifile,
                    file=file,
                    confidence=confidence,
                    isyn=isyn,
                    label=label,
                    ilabel=ilabel,
                    path=path,
                    image_width=image_width,
                    image_height=image_height,
                ))
                future = threads.submit(
                    submit,
                    result,
                    outpath,
                    *args,
                    **kwargs
                )
                futures.append(future)
                # yielding future allows you to control the rate
                yield future

            for future in futures:
                future.result()


class Owl(AutoDistill):
    @cached_property
    def model(self) -> type[OWLv2]:
        from autodistill_owlv2 import OWLv2
        return OWLv2


class GDino(AutoDistill):
    @cached_property
    def model(self) -> type[GroundingDINO]:
        from autodistill_grounding_dino import GroundingDINO
        return GroundingDINO



class Predict(magic.Magic):
    __outer__: Elsa
    gdino = GDino()
    owl = Owl()
    gdino3p = GDino3P()

    @magic.cached.property
    def ontologies(self) -> list[CaptionOntology]:
        prompts: Series[str] = self.__outer__.truth.prompts.natural
        ontologies = [
            CaptionOntology({prompt: prompt})
            for prompt in prompts
        ]
        return ontologies


if __name__ == '__main__':
    from elsa.root import Elsa
    from elsa.files import unified_dhodcz

    elsa: Elsa = Elsa.from_unified(files=unified_dhodcz)

    # pass boolean masks to control which files and prompts to use
    files = elsa.files.nfile < 5
    # prompts = elsa.truth.prompts.iprompt < 5
    # prompts = elsa.truth.unique.rephrase.prompts.isyns < 5
    prompts = elsa.prompts

    # files = elsa.files.ifile == 'BSV_7'
    # prompts = elsa.truth.unique.rephrase.prompts.natural == 'an individual on a phone strolling'
    # elsa.predict.gdino(prompts=prompts)
    # 'a person' in prompts.natural.values
    # prompts.natural.str.contains('person')
    prompts = None

    futures = elsa.predict.gdino(files=files, prompts=prompts)
    # # here we do nothing with the futures and just let the function run
    for future in futures:
        ...

    # futures = elsa.predict.owl(files=files, prompts=prompts)
    # for future in futures:
    #     ...

    # futures = elsa.predict.gdino(files=files, prompts=prompts)
    # here we print each result as it's completed serially
    for future in futures:
        print(future.result())
    #
    # futures = elsa.predict.gdino(files=files, prompts=prompts)
    # # here we only run the first inference and print the path
    # future = next(futures)
    # print(future.result())
    #

    """
    predictions = [
        batch()
        for batch in elsa.predict.owl(outdir=...)
    ]
    Elsa.from_unified(predictions=predictions)

    for batch in elsa.predict.owl(outdir=...):
        batch()
    """

    ...
