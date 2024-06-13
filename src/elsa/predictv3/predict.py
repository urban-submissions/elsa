from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import functools
import groundingdino.datasets.transforms as T
import numpy as np
import pandas as pd
import tempfile
import torch
import torchvision.transforms.functional as F
import tqdm
from PIL import Image
from dataclasses import dataclass
from dataclasses import field
from functools import *
from pandas import DataFrame
from pandas import MultiIndex
from pandas import Series
from pathlib import Path
from tensorflow import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import *

# please make sure https://github.com/IDEA-Research/GroundingDINO is installed correctly.
import magicpandas as magic
from elsa.annotation.prompts import Prompts
from elsa.files import Files
from elsa.logits.logits import Logits
from elsa.predictv3.selection import Selection


@dataclass
class PromptIteration:
    prompt: Prompts
    logits: Logits = field(repr=False)
    outpath: Path = field(repr=False)

    def save_logits(self, frame, path: Path):
        frame.attrs['prompt'] = self.prompt.natural
        # noinspection PyTypeChecker
        frame.attrs['isyns'] = ','.join(map(str, self.prompt.isyns))
        path.parent.mkdir(parents=True, exist_ok=True)
        it = map('.'.join, frame.columns)
        columns = pd.Index(it)
        frame = frame.set_axis(columns, axis=1)
        _ = (
            frame
            .set_axis(columns, axis=1)
            .to_parquet(path)
        )


def interpolate_transform(images: Files):
    del images.batch_width
    del images.batch_height
    del images.width_over_batch_width
    del images.height_over_batch_height
    size = images.batch_height, images.batch_width

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    interpolate = functools.partial(
        torch.nn.functional.interpolate,
        size=size,
        mode='bilinear',
        align_corners=False,
    )

    # see groundingdino.util.inference:38
    def wrapper(image):
        IMAGE = image
        image = IMAGE
        image = image.convert("RGB")
        image = F.to_tensor(image)
        # as we are not training randomresize may not be necessary?
        image = interpolate(image[None])[0]
        image = F.normalize(image, mean, std)
        image = image.to('cuda')
        return image

    return wrapper


@dataclass
class ImageIteration:
    selection: Selection = field(repr=False)
    file: str
    prompt: str

    def view_file(self):
        print(self.file)
        image = Image.open(self.file)
        return image


class ImageDataset(Dataset):
    def __init__(self, file_paths: Files, transforms):
        self.paths: Files = file_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    @cached_property
    def _paths(self) -> Series[str]:
        return self.paths.path.values

    def __getitem__(self, idx):
        path = self._paths[idx]
        image = Image.open(path)
        if self.transforms:
            image = self.transforms(image)
        return image, path


class ImageLoader:
    def __init__(
            self,
            files: Files,
            batch_size: int,
    ):
        """Depending on batch size map each shape to a list of batches"""
        self.files = files
        self.batch_size = batch_size

        def apply(frame: DataFrame):
            return [
                frame.iloc[i:i + batch_size]
                for i in range(0, len(frame), batch_size)
            ]

        self.shape2list_files: dict[tuple, list[Files]] = (
            files
            .groupby('height width'.split(), sort=False)
            .apply(apply)
            .to_dict()
        )

    @cached_property
    def transform(self):
        """See groundingdino.util.load_image"""
        transforms = [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
        transform = T.Compose(transforms)
        return transform

    @property
    def lists(self) -> Iterator[list[Image.Image]]:
        """Each iteration yields a list of Images"""
        shapes_files: Iterator[tuple[tuple, Files]] = (
            (shape, files)
            for shape, list_files
            in self.shape2list_files.items()
            for files in list_files
        )
        with ThreadPoolExecutor() as threads:
            it = (
                threads.map(Image.open, files.path.values)
                for shape, files in shapes_files
            )
            # preemptive loading of the next image
            prev = next(it)
            while True:
                try:
                    curr = next(it)
                except StopIteration:
                    break
                yield list(prev)
                prev = curr
            yield list(prev)

    def __iter__(self) -> Iterator[tuple[Tensor, Files]]:
        """Each iteration yields an (N, 3, H, W) tensor and the files"""
        it_files = (
            files
            for shape, files
            in self.shape2list_files.items()
            for files in files
        )
        lists = self.lists
        it = zip(it_files, lists)
        transform = self.transform
        for files, images in it:
            images = [
                transform(image.convert("RGB"), target=None)[0]
                for image in images
            ]
            images = torch.stack(images).cuda()
            yield images, files


class GDino3P(magic.Magic):
    if False:
        from ..predict import Predict
    # __outer__: Predict
    __outer__: Elsa

    @magic.cached.cmdline.property
    def box_threshold(self):
        return .3

    @magic.cached.cmdline.property
    def text_threshold(self):
        return .25

    @magic.cached.cmdline.property
    def device(self):
        # cuda or cpu
        return "cuda"

    @magic.cached.cmdline.property
    def cpu_only(self) -> bool:
        return False

    def transform(self, images: Files):
        del images.batch_width
        del images.batch_height
        del images.width_over_batch_width
        del images.height_over_batch_height
        size = images.batch_height, images.batch_width

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        interpolate = functools.partial(
            torch.nn.functional.interpolate,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        # see groundingdino.util.inference:38
        def wrapper(image):
            IMAGE = image
            image = IMAGE
            image = image.convert("RGB")
            image = F.to_tensor(image)
            # as we are not training randomresize may not be necessary?
            image = interpolate(image[None])[0]
            image = F.normalize(image, mean, std)
            image = image.to(self.device)
            return image

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        def wrapper(image):
            image = image.convert("RGB")
            image = transform(image, target=None)[0]
            image = image.cuda()
            return image

        return wrapper

    def images(self, files: Series, transform) -> Iterator[Image]:
        """Preemptively loads the next image"""
        with ThreadPoolExecutor() as threads:
            files = threads.map(Image.open, files)
            prev = next(files)
            while True:
                try:
                    curr = next(files)
                except StopIteration:
                    break
                yield transform(prev)
                prev = curr
            yield transform(prev)

    def batched_without_interpolation(
            self,
            config=None,
            checkpoint=None,
            outdir: str | Path = None,
            batch_size: int = None,
            files=None,
            prompts=None,
            force=False,
            nested=True,
    ) -> Iterator[ImageIteration | PromptIteration]:
        from elsa.predictv3 import gdino
        if config is None:
            config = gdino.LocalFiles.config
        if config is None:
            raise ValueError('config must be provided')
        if checkpoint is None:
            checkpoint = gdino.LocalFiles.checkpoint
        if checkpoint is None:
            raise ValueError('checkpoint must be provided')
        if batch_size is None:
            batch_size = gdino.LocalFiles.batch_size
        if batch_size is None:
            raise ValueError('batch_size must be provided')
        if files is None:
            files = slice(None)
        if prompts is None:
            prompts = slice(None)
        if outdir is None:
            outdir = Path(tempfile.mkdtemp())
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        empty = None
        # elsa = self.__outer__.__outer__
        elsa = self.__outer__
        truth = elsa.truth
        PROMPTS = elsa.prompts
        _ = PROMPTS.isyns
        FILES = elsa.files
        _ = (
            PROMPTS.isyns, PROMPTS.natural, elsa.labels.isyn,
            PROMPTS.global_ilast, PROMPTS.ilast, PROMPTS.ifirst,
            PROMPTS.isyns, PROMPTS.cardinal,
            truth.path, FILES.nfile, FILES.path, FILES.ifile,
            FILES.width_over_batch_width, FILES.height_over_batch_height,
        )

        prompts = PROMPTS.loc[prompts]
        model = (
            gdino.GroundingDINO
            .from_elsa(config, checkpoint)
            .cuda()
        )
        paths = FILES.path.loc[files]
        loader = ImageLoader(FILES.loc[files], batch_size)

        message = (
            f'Using {len(prompts)} prompts to predict {len(paths)} images '
            f'with {self.__class__.__name__}'
        )
        self.__logger__.info(message)
        it = (
            prompts
            .sort_values('isyns')
            .reset_index()
            .itertuples(index=False)
        )

        counter = tqdm.tqdm(total=len(prompts))
        for prompt in it:  # TODO: group by synonyms here
            natural = prompt.natural
            cardinal: str = prompt.cardinal
            if nested:
                outpath = Path(
                    outdir,
                    cardinal,
                    f'{natural}.parquet',
                ).resolve()
            else:
                outpath = Path(
                    outdir,
                    f'{natural}.parquet',
                ).resolve()
            if (
                    not force
                    and outpath.exists()
            ):
                continue

            list_confidence = []
            list_xywh = []
            captions = [prompt.natural + '.']  # gdino requires .
            prev = None
            offset_mapping = None
            icol = NotImplementedError
            list_file = []
            list_path = []
            list_ifile = []

            for images, files in loader:

                # counter.update(images.shape[0])
                assert len(images.shape) == 4, f"wrong number of axis per image, forgot to batch? {images.shape}"
                with torch.no_grad():
                    outputs: gdino.Result = model(images, captions=captions * images.shape[0])  # replicating the caption batch_size times

                offset_mapping = (
                    outputs.offset_mapping
                    .squeeze()
                    .cpu()
                    .numpy()
                    # because we passed the same caption multiple times,
                    # we can just access the first; if multiple different
                    # captions, we will have to change this
                )
                if len(offset_mapping.shape) == 3:
                    offset_mapping = offset_mapping[0]
                if prev is None:
                    icol = np.arange(len(offset_mapping))
                    loc = offset_mapping[:, 1] > 0
                    loc &= offset_mapping[:, 0] < (len(natural) - 1)
                    offset_mapping = offset_mapping[loc]
                    icol = icol[loc]
                elif not np.all(offset_mapping == prev):
                    raise ValueError('offset_mapping not consistent')

                confidence = (
                    outputs.pred_logits
                    .sigmoid()
                    [:, :, icol]
                    .cpu()
                    .numpy()
                )
                list_confidence += list(confidence)  # unrolling batch
                xywh = (
                    outputs.pred_boxes
                    .cpu()
                    .numpy()
                )
                list_xywh += list(xywh)  # unrolling batch...

                repeat = np.fromiter(map(len, xywh), dtype=int)
                list_file += list(files.file.values.repeat(repeat))
                list_path += list(files.path.values.repeat(repeat))
                list_ifile += list(files.ifile.values.repeat(repeat))

            names = 'span ifirst'.split()
            file = np.array(list_file)
            path = np.array(list_path)
            xywh = np.concatenate(list_xywh)
            colnames = (
                'normx normy normwidth normheight file path prompt'
            ).split()
            empty = [''] * len(colnames)
            arrays = colnames, empty,
            columns = MultiIndex.from_arrays(arrays, names=names)
            try:
                data = np.concatenate(list_xywh)
            except ValueError:
                if (
                        not force
                        and outpath.exists()
                ):
                    continue
                if empty is None:
                    empty = Logits(columns=columns)
                    empty.attrs = {}
                iteration = PromptIteration(prompt, empty, outpath)
                yield iteration
                continue

            _prompt = np.full(len(data), natural, dtype=f'U{len(natural)}')
            a = xywh
            data = a[:, 0], a[:, 1], a[:, 2], a[:, 3], file, path, _prompt
            xywh = DataFrame(dict(zip(columns, data)))

            span = np.fromiter((
                natural[ifirst:ilast]
                for ifirst, ilast in offset_mapping
            ), dtype=object, count=offset_mapping.shape[0])
            ifirst = offset_mapping[:, 0].astype(str)
            arrays = span, ifirst
            columns = MultiIndex.from_arrays(arrays, names=names)
            data = np.concatenate(list_confidence, axis=0)
            confidence = DataFrame(data, columns=columns)
            result = pd.concat([confidence, xywh], axis=1).pipe(Logits)
            result['file'] = result['file'].astype('category')
            result['path'] = result['path'].astype('category')
            iteration = PromptIteration(prompt, result, outpath)
            result.attrs = {}

            yield iteration
            counter.update()

    def __call__(
            self,
            config=None,
            checkpoint=None,
            outdir: str | Path = None,
            extension: str = 'parquet',
            files=None,
            prompts=None,
            force=False,
            synonyms=False,
            yield_image=False,
            yield_prompt=True,
            *args,
            **kwargs,
    ) -> Iterator[ImageIteration | PromptIteration]:
        from elsa.predictv3 import gdino
        if config is None:
            config = gdino.LocalFiles.config
        if checkpoint is None:
            checkpoint = gdino.LocalFiles.checkpoint
        if files is None:
            files = slice(None)
        if prompts is None:
            prompts = slice(None)
        if outdir is None:
            outdir = Path(tempfile.mkdtemp())
        batch_size = gdino.LocalFiles.batch_size
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        empty = None
        elsa = self.__outer__.__outer__
        truth = elsa.truth
        PROMPTS: Prompts = truth.unique.rephrase.prompts
        REPHRASE = truth.unique.rephrase
        FILES = elsa.files
        _ = (
            PROMPTS.isyns, PROMPTS.natural, elsa.labels.isyn,
            PROMPTS.global_ilast, PROMPTS.ilast, PROMPTS.ifirst,
            PROMPTS.isyns, PROMPTS.cardinal,
            truth.path, FILES.nfile, FILES.path, FILES.ifile,
            FILES.width_over_batch_width, FILES.height_over_batch_height,
        )
        if synonyms:
            # select all prompts synonymous with current prompts
            isyns = PROMPTS.isyns.loc[prompts]
            loc = PROMPTS.isyns.isin(isyns).values
            prompts |= loc

        prompts = PROMPTS.loc[prompts]
        model = (
            gdino.GroundingDINO
            .from_elsa(config, checkpoint)
            .cuda()
        )
        paths = FILES.path.loc[files]
        images = IMAGES = FILES.loc[files]
        files = FILES.files.loc[files]

        message = (
            f'Using {len(prompts)} prompts to predict {len(paths)} images '
            f'with {self.__class__.__name__}'
        )
        self.__logger__.info(message)
        it = (
            prompts
            .sort_values('isyns')
            .reset_index()
            .itertuples(index=False)
        )

        # total = len(prompts) * len(paths) // batch_size
        # should not be divided by batch size because number of files per iteration can be <= batch_size
        total = len(prompts) * len(paths)
        counter = tqdm.tqdm(total=total, desc=message, bar_format='{elapsed}<{remaining}')

        # creating image loader once for all prompts
        # transform = self.transform(images)
        transform = interpolate_transform(images)
        img_ds = ImageDataset(files, transform)
        img_loader = DataLoader(img_ds, batch_size=batch_size, shuffle=False)

        # TODO: load the split classes here along with the natural prompts,
        # we must have a list that gives each component of the natural prompt separated ["an individual", "walking"]
        # so that we can easily link the N tokens to the M words in the natural prompts where M!=N

        # @marco changed so if user passes synonyms, all synonymous prompts are included.
        #   synonymous prompts go into same subdirectory named with "cardinal" prompt

        for prompt in it:  # TODO: group by synonyms here
            natural = prompt.natural
            cardinal: str = prompt.cardinal
            outpath = Path(
                outdir,
                cardinal,
                f'{natural}.{extension}',
            ).resolve()
            if (
                    not force
                    and outpath.exists()
            ):
                continue

            list_confidence = []
            list_xywh = []
            captions = [prompt.natural + '.']  # gdino requires .
            prev = None
            offset_mapping = None
            icol = NotImplementedError

            for images, _ in img_loader:
                counter.update(images.shape[0])
                assert len(images.shape) == 4, f"wrong number of axis per image, forgot to batch? {images.shape}"
                with torch.no_grad():
                    outputs: gdino.Result = model(images, captions=captions * images.shape[0])  # replicating the caption batch_size times

                offset_mapping = (
                    outputs.offset_mapping
                    .squeeze()
                    .cpu()
                    .numpy()
                    # because we passed the same caption multiple times,
                    # we can just access the first; if multiple different
                    # captions, we will have to change this
                )
                if len(offset_mapping.shape) == 3:
                    offset_mapping = offset_mapping[0]
                if prev is None:
                    icol = np.arange(len(offset_mapping))
                    loc = offset_mapping[:, 1] > 0
                    loc &= offset_mapping[:, 0] < (len(natural) - 1)
                    offset_mapping = offset_mapping[loc]
                    icol = icol[loc]
                elif not np.all(offset_mapping == prev):
                    raise ValueError('offset_mapping not consistent')

                confidence = (
                    outputs.pred_logits
                    .sigmoid()
                    [:, :, icol]
                    .cpu()
                    .numpy()
                )
                list_confidence += list(confidence)  # unrolling batch
                xywh = (
                    outputs.pred_boxes
                    .cpu()
                    .numpy()
                )
                list_xywh += list(xywh)  # unrolling batch

            names = 'meta label span ifirst'.split()

            repeat = list(map(len, list_xywh))
            file = files.file.values.repeat(repeat)
            path = files.path.values.repeat(repeat)
            ifile = files.ifile.values.repeat(repeat)
            colnames = (
                'normx normy normwidth normheight file path prompt'
            ).split()
            arrays = (
                colnames,
                [''] * len(colnames),
                [''] * len(colnames),
                [''] * len(colnames),
            )
            columns = MultiIndex.from_arrays(arrays, names=names)
            try:
                data = np.concatenate(list_xywh)
            except ValueError:
                if (
                        not force
                        and outpath.exists()
                ):
                    continue
                if empty is None:
                    empty = Logits(columns=columns)
                iteration = PromptIteration(prompt, empty, outpath)
                yield iteration
                continue

            _prompt = np.full(len(data), natural, dtype=f'U{len(natural)}')
            arrays = data, file[:, None], path[:, None], _prompt[:, None]
            data = np.concatenate(arrays, axis=1)
            xywh = DataFrame(data, columns=columns)

            # ratio = IMAGES.ratio / IMAGES.batch_ratio
            # original 4:3  1.3
            # batch 16:9    1.7

            # afterwards, go from 1.7 back to 1.3
            # min / max * smaller dimension
            # 1.3 / 1.7 * normheight

            # if ratio < batch_ratio:   image is taller
            #   width *= ratio / bratio
            # else:                     image is wider
            #   height *= bratio / ratio

            ratio = IMAGES.ratio.loc[ifile].values
            batch_ratio = IMAGES.batch_ratio

            loc = ratio < batch_ratio

            # where image is taller, scale down width
            xywh.loc[loc, 'normwidth'] = (
                    xywh.normwidth.loc[loc] * ratio[loc] / batch_ratio
            ).values
            xywh.loc[loc, 'normx'] = (
                    xywh.normx.loc[loc] * ratio[loc] / batch_ratio
            ).values

            # # where image is wider, scale down height
            xywh.loc[~loc, 'normheight'] = (
                    xywh.normheight.loc[~loc] * batch_ratio / ratio[~loc]
            ).values
            xywh.loc[~loc, 'normy'] = (
                    xywh.normy.loc[~loc] * batch_ratio / ratio[~loc]
            ).values

            ilast = offset_mapping[:, 0] + prompt.global_ilast
            iloc = np.searchsorted(REPHRASE.global_ilast, ilast)
            label = REPHRASE.natural.iloc[iloc].values
            meta = REPHRASE.meta.iloc[iloc].values
            span = np.fromiter((
                natural[ifirst:ilast]
                for ifirst, ilast in offset_mapping
            ), dtype=object, count=len(iloc))
            ifirst = offset_mapping[:, 0].astype(str)
            arrays = meta, label, span, ifirst
            columns = MultiIndex.from_arrays(arrays, names=names)
            assert all(
                isinstance(ifirst, str)
                for ifirst in columns.get_level_values('ifirst')
            )
            data = np.concatenate(list_confidence, axis=0)
            confidence = DataFrame(data, columns=columns)
            result = pd.concat([confidence, xywh], axis=1).pipe(Logits)
            result['file'] = result['file'].astype('category')
            result['path'] = result['path'].astype('category')
            iteration = PromptIteration(prompt, result, outpath)

            assert (columns.dtypes == object).all()
            assert (result.normx <= 1).all()
            assert (result.normy <= 1).all()

            yield iteration


if __name__ == '__main__':
    from elsa import Elsa

    elsa = Elsa.from_bing()
    for iteration in elsa.predict.gdino3p():
        ...
