from __future__ import annotations

import logging
import numpy as np
import os
import pandas as pd
import tqdm
# from autodistill.detection import CaptionOntology
from functools import cached_property
from numpy import ndarray
from pandas import Series

import magicpandas as magic
from elsa.predictions import Predictions

if False:
    from .root import Elsa
    from autodistill_grounding_dino import GroundingDINO
    from autodistill_owlv2 import OWLv2
CaptionOntology = {}

class AutoDistill(magic.Magic):
    model: GDino | OWLv2
    __outer__: Predict

    def one_at_a_time(
            self,
            quiet=False,
    ) -> Predictions:
        """Predict using one CaptionOntology per unique prompt"""

        import tensorflow as tf
        if quiet:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            logging.getLogger('transformers').setLevel(logging.ERROR)
            logging.getLogger('PIL').setLevel(logging.ERROR)

        elsa = self.__outer__.__outer__
        truth = elsa.truth
        prompts = truth.unique.rephrase.prompts
        loc = ~truth.ifile.duplicated()
        _ = truth.path, truth.nfile

        truth = truth.loc[loc]
        wsen: list[ndarray] = []
        confidence: list[ndarray] = []

        total = len(truth) * len(prompts)
        message = (
            f'Using {len(prompts)} prompts to predict {len(truth)} images '
            f'with {self.model.__class__.__name__}...'
        )
        self.__logger__.info(message)
        pbar = tqdm.tqdm(total=total)
        for ontology in prompts.ontology:
            model: GroundingDINO | OWLv2 = self.model(ontology=ontology)
            for path in truth.path:
                detect = model.predict(path)
                wsen.append(detect.xyxy)
                confidence.append(detect.confidence)
                pbar.update(1)

            # # simulate prediction
            # for path in truth.path:
            #     wsen.append([])
            #     confidence.append([])
            #     pbar.update(1)

        repeat = REPEAT = np.fromiter(map(len, wsen), int, len(wsen))
        wsen: ndarray = np.concatenate(wsen)
        confidence: ndarray = np.concatenate(confidence)

        # the following code broadcast data to all align for a dataframe
        natural = (
            prompts.natural.values
            .repeat(len(truth))
            .repeat(REPEAT)
        )
        ifile = (
            np.tile(truth.ifile, len(prompts))
            .repeat(REPEAT)
        )
        isyns = (
            prompts.isyns.values
            .repeat(len(truth))
            .repeat(REPEAT)
        )
        repeat = (
            prompts.__outer__.__outer__
            .groupby('isyns')
            .size()
            .loc[isyns]
        )
        isyn = (
            prompts.__outer__.__outer__.isyn
            .loc[isyns]
            .values
        )
        iloc = np.arange(len(repeat)).repeat(repeat)
        _ = elsa.labels.isyn
        ilabel = (
            elsa.labels
            .reset_index()
            .drop_duplicates('isyn')
            .set_index('isyn')
            .ilabel
            .loc[isyn]
            .values
        )
        assert len(iloc) == len(ilabel)

        columns = 'w s e n'.split()
        if not len(wsen):
            wsen = None
        result = (
            Predictions(wsen, columns=columns)
            .assign(
                ifile=ifile,
                confidence=confidence,
                natural=natural,
            )
            .iloc[iloc]
            .assign(
                isyn=isyn,
                ilabel=ilabel,
            )
            .indexed()
        )
        result: Predictions = getattr(elsa.__class__, 'predictions').__second__(result)
        result.__outer__ = elsa
        result.__owner__ = elsa
        result.__first__ = elsa.__class__.predictions
        _ = result.label, result.iann
        return result

    def one_group_of_synonymous_labels_at_a_time(self):
        """Predict using one CaptionOntology per unique isyns tuple"""
        raise NotImplementedError

    # todo: hwo to map isyns back to labels?
    def all_at_once(self) -> Predictions:
        """Predict using only one CaptionOntology"""
        raise NotImplementedError
        elsa = self.__outer__.__outer__
        truth = elsa.truth
        prompts = truth.prompts
        ontology = prompts.ontologies.one
        model: GDino | OWLv2 = self.model(ontology=ontology)
        loc = ~truth.ifile.duplicated()
        _ = truth.path
        truth = truth.loc[loc]
        wsen: list[ndarray] = []
        irephrase: list[ndarray] = []
        confidence: list[ndarray] = []
        for path in truth.path:
            detect = model.predict(path)
            wsen.append(detect.xyxy)
            confidence.append(detect.confidence)
            irephrase.append(detect.class_id)

        repeat = np.fromiter(map(len, wsen), int, len(wsen))
        wsen: ndarray = np.concatenate(wsen)
        irephrase: ndarray = np.concatenate(irephrase)
        ifile = truth.ifile.repeat(repeat)
        repeat = (
            prompts.ilabel
            .groupby('irephrase')
            .size()
            .loc[irephrase]
        )
        iloc = np.arange(len(repeat)).repeat(repeat)
        ilabel = prompts.ilabel.loc[irephrase].values
        iann = np.arange(repeat.sum())
        columns = 'w s e n'.split()
        result = (
            Predictions(wsen, columns=columns)
            .assign(ifile=ifile)
            .iloc[iloc]
            .assign(
                ilabel=ilabel,
                iann=iann,
            )
            .indexed()
        )
        result.__outer__ = elsa
        result.__owner__ = elsa
        return result

    # whichever turns out to be the best, assign to call
    __call__ = one_at_a_time


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

    @magic.cached.property
    def ontologies(self) -> list[CaptionOntology]:
        prompts: Series[str] = self.__outer__.truth.prompts.natural
        ontologies = [
            CaptionOntology({prompt: prompt})
            for prompt in prompts
        ]
        return ontologies


