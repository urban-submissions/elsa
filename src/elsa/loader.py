from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import PIL
import pandas as pd
import torch
from PIL import Image
from functools import cached_property
from torch.utils.data import Dataset
from torchvision import transforms
from typing import *

import magicpandas as magic
from elsa.files import Files

if False:
    # from .combos import Combos
    import combos
    from .combos import Combos
    from .root import Elsa


class Loaded(NamedTuple):
    image: torch.Tensor
    ibox: torch.Tensor


class Loader(
    torch.utils.data.Dataset,
    magic.Magic
):
    __outer__: Combos

    def __getitem__(self, item: int) -> Loaded:
        """ Pass an index; get image and ibox """
        path = self.paths[item]
        image = (
            PIL.Image
            .open(path)
            .convert('RGB')
        )
        image = self.transform(image)
        frame = self.frames[item]
        ibox = frame.ibox.values
        result = Loaded(image, ibox)
        # note: passed ibox; use predictions.subloc[ibox] to get frame
        return result

    def ibox2data(
            self,
            ibox: torch.Tensor,
            columns: tuple[str] = ('normx', 'normy', 'normwidth', 'normheight', 'label')
    ) -> Combos:
        if columns is None:
            columns = slice(None)
        """Pass the ibox from loader[...] to get YOLO data format"""
        loc = pd.Series(ibox, name='ibox')
        result = self.combos.subloc[loc, columns]
        return result

    def __len__(self):
        return len(self.files)

    @magic.cached.base.property
    def frames(self) -> list[Combos]:
        """A list of frames for each file to be used in __getitem__"""
        c = self.combos
        _ = c.normx, c.normy, c.normwidth, c.normheight, c.label, c.ifile
        result: list[Combos | pd.DataFrame] = [
            frame
            for _, frame
            in
            self.combos
            .reset_index()
            .groupby('ifile', sort=False)
        ]
        return result

    @Files
    def files(self):
        # get the first combo for each iifle
        ifile = (
            self.combos
            .reset_index()
            .ifile
            .drop_duplicates()
        )
        result = self.__outer__.files.loc[ifile]
        return result

    @property
    def images(self) -> Iterator[PIL.Image.Image]:
        load = PIL.Image.open
        with ThreadPoolExecutor() as threads:
            for image in threads.map(load, self.paths):
                image = image.convert('RGB')
                yield self.transform(image)

    @magic.cached.base.property
    def paths(self) -> list[str]:
        return self.files.path.tolist()

    @magic.cached.base.property
    def ifiles(self) -> list[int]:
        return self.files.ifile.tolist()

    @magic.cached.cmdline.property
    def mode(self) -> str:
        """
        CS filters out activities and others from ground truth,
        CSA filters others and removes CS only labels
        CSAT keeps only annotations with all the categories
        """

    @magic.cached.cmdline.property
    def th(self) -> int:
        """
        min amount of occurrences needed to use that combo_label
        (currently ineffective)
        """

    @th.setter
    def th(self, value: int):
        if value <= 0:
            raise ValueError("th must be greater than 0")
        return value

    @magic.cached.cmdline.property
    def res(self) -> int:
        """ image resolution from the dataloader """
        return 256

    @magic.cached.base.property
    def transform(self) -> Callable[[...], torch.Tensor]:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
        size = self.res, self.res
        interpolation = Image.BICUBIC
        resize = transforms.Resize(size, interpolation)
        totensor = transforms.ToTensor()
        result = transforms.Compose([resize, totensor, normalize])
        return result

    @magic.cached.frame.property
    def combos(self) -> Combos:
        truth = self.__outer__.truth.combos
        iloc = truth.nfile.argsort()
        result = truth.iloc[iloc]
        result = result.copy()
        return result


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('error', message='.*reduce_op.*', category=UserWarning)

    import os.path
    from elsa import Elsa

    files = '/home/arstneio/PycharmProjects/elsa/src/elsa/static/bing/files'
    if not os.path.exists(files):
        files = None
    elsa = Elsa.from_bing(files=files)
    loader = elsa.predictions.loader
    len(loader)
    # returns img, ibox
    img, ibox = loader[0]
    ibox = pd.Series(ibox, name='ibox')  # subloc needs the name
    pred = elsa.predictions.subloc[ibox]
    pred
