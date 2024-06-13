from __future__ import annotations

from collections import Counter
from collections import UserDict

import geopandas as gpd
import numpy as np
import os
import pandas as pd
from dataclasses import dataclass
from functools import cached_property
from numpy import ndarray
from pandas import Series, Index
from pathlib import Path
from shapely import *
from typing import *

from elsa import boxes

if False:
    from elsa.annotation.annotation import Annotated

colors = (
    'red orange green blue purple '
    'brown pink gray grey cyan magenta lime navy teal '
    'gold silver violet indigo maroon olive fuchsia '
    'aquamarine turquoise khaki lavender tan coral salmon '
    'sienna beige plum wheat orchid tomato yellowgreen '
    'seagreen skyblue lightblue powderblue '
    'royalblue mediumblue azure chartreuse mediumseagreen '
    'springgreen palegreen mediumspringgreen lawngreen '
    'lightgreen darkgreen forestgreen limegreen '
    'greenyellow aqua deepskyblue dodgerblue steelblue '
    'slateblue lightskyblue lightseagreen darkcyan '
    'cadetblue darkturquoise mediumturquoise paleturquoise '
    'darkslateblue midnightblue cornflowerblue lightslategray '
    'slategray lightslategrey slategrey '
    'lightsteelblue mediumslateblue lightgray '
    'lightgrey gainsboro '
    'honeydew mintcream aliceblue seashell '
    'oldlace ivory '
    'lavenderblush mistyrose'
).split()

colors2 = (
    'red orange green blue purple'
    'brown pink gray grey cyan magenta lime navy teal '
    'gold silver violet indigo maroon olive fuchsia '
    'aquamarine turquoise khaki lavender tan coral salmon '
    'sienna beige plum wheat orchid tomato yellowgreen '
    'seagreen skyblue lightblue powderblue '
    'royalblue mediumblue azure chartreuse mediumseagreen '
    'springgreen palegreen mediumspringgreen lawngreen '
    'lightgreen darkgreen forestgreen limegreen '
    'greenyellow aqua deepskyblue dodgerblue steelblue '
    'slateblue lightskyblue lightseagreen darkcyan '
    'cadetblue darkturquoise mediumturquoise paleturquoise '
    'darkslateblue midnightblue cornflowerblue lightslategray '
    'slategray lightslategrey slategrey '
    'lightsteelblue mediumslateblue lightgray '
    'lightgrey gainsboro '
    'honeydew mintcream aliceblue seashell '
    'oldlace ivory '
    'lavenderblush mistyrose'
).split()

@dataclass
class Constituents:
    unique: ndarray
    ifirst: ndarray
    ilast: ndarray
    istop: ndarray
    repeat: ndarray

    @cached_property
    def indices(self) -> ndarray:
        return np.arange(len(self)).repeat(self.repeat)

    def __len__(self):
        return len(self.unique)

    def __repr__(self):
        return f'Constituents({self.unique}) at {hex(id(self))}'

    def __getitem__(self, item) -> Constituents:
        unique = self.unique[item]
        ifirst = self.ifirst[item]
        ilast = self.ilast[item]
        istop = self.istop[item]
        repeat = self.repeat[item]
        con = Constituents(unique, ifirst, ilast, istop, repeat)
        return con


def constituents(self: Union[Series, ndarray, Index], monotonic=True) -> Constituents:
    try:
        monotonic = self.is_monotonic_increasing
    except AttributeError:
        pass
    if monotonic:
        if isinstance(self, (Series, Index)):
            assert self.is_monotonic_increasing
        elif isinstance(self, ndarray):
            assert np.all(np.diff(self) >= 0)

        unique, ifirst, repeat = np.unique(self, return_counts=True, return_index=True)
        istop = ifirst + repeat
        ilast = istop - 1
        # constituents = Constituents(unique, ifirst, ilast, istop, repeat)
        constituents = Constituents(
            unique=unique,
            ifirst=ifirst,
            ilast=ilast,
            istop=istop,
            repeat=repeat,
        )
    else:
        counter = Counter(self)
        count = len(counter)
        repeat = np.fromiter(counter.values(), dtype=int, count=count)
        unique = np.fromiter(counter.keys(), dtype=self.dtype, count=count)
        val_ifirst: dict[int, int] = dict()
        val_ilast: dict[int, int] = {}
        for i, value in enumerate(self):
            if value not in val_ifirst:
                val_ifirst[value] = i
            val_ilast[value] = i
        ifirst = np.fromiter(val_ifirst.values(), dtype=int, count=count)
        ilast = np.fromiter(val_ilast.values(), dtype=int, count=count)
        istop = ilast + 1
        constituents = Constituents(unique, ifirst, ilast, istop, repeat)

    return constituents


import PIL.Image
import PIL.ImageDraw


def view_detection(detections, image: str) -> PIL.Image:
    # Load the image
    result = PIL.Image.open(image)
    draw = PIL.ImageDraw.Draw(result)
    color_map = {
        i: colors[i]
        for i in set(detections.class_id)
    }
    print(color_map)

    # Iterate over detections and draw each one
    for box, class_id in zip(detections.xyxy, detections.class_id):
        # Convert box coordinates to integer, since draw.rectangle expects integer tuples
        box = tuple(map(int, box))
        # Fetch the color corresponding to the class_id from util.colors
        color_map = colors[class_id]

        # Draw the rectangle
        draw.rectangle(box, outline=color_map)

    return result


def get_ibox(
        boxes: Annotated,
        label=False,
        round=4,
) -> Series[int]:
    _ = boxes.w, boxes.s, boxes.e, boxes.n, boxes.file
    columns = 'file w s e n'.split()
    if label:
        _ = boxes.isyn
        columns.append('isyn')
    needles = boxes[columns].copy()
    columns = 'w s e n'.split()
    # round the columns w s e n to 4 decimal places
    needles[columns] = needles[columns].round(round)
    needles = needles.pipe(pd.MultiIndex.from_frame)
    haystack = needles.unique()
    ibox = (
        pd.Series(np.arange(len(haystack)), index=haystack)
        .loc[needles]
    )
    return ibox


T = TypeVar('T')


def trim_path(path: T) -> T:
    """Removes the directories and the extension from a path."""
    if isinstance(path, Series):
        result = (
            pd.Series(path)
            .str
            .rsplit(os.sep, expand=True, n=1)
            .iloc[:, -1]
            .str
            .split('.', expand=True, n=1)
            .iloc[:, 0]
            .values
        )
    elif isinstance(path, str):
        result = (
            str(path)
            .rsplit(os.sep, 1)[-1]
            .split('.', 1)[0]
        )
    elif isinstance(path, Path):
        result = (
            Path(path)
            .name
            .split('.', 1)[0]
        )
    else:
        raise TypeError(f'path must be Series, str or Path, not {type(path)}')
    return result


class LocalFile(UserDict):
    def __get__(self, instance: LocalFiles, owner) -> Optional[str]:
        key = os.environ.get('sirius')
        return self.get(key)


class LocalFiles:
    """
    Allows user to map users to local file maps. Here, if os.env[elsa]
    is dhodcz2, LocalFiles.config returns the respective path.
    This way we can better share scripts we're working on.

    class LocalFiles(LocalFiles_):
        config: str = dict(
            dhodcz2='/home/arstneio/PycharmProjects/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py',
            marco="/home/mcipriano/projects/elsa/configs/cfg_odvg.py"
        )

    To achieve this, I used
    echo 'export elsa="dhodcz2"' >> venv/bin/activate && source venv/bin/activate
    """

    def __init_subclass__(cls, **kwargs):
        for key, value in cls.__dict__.items():
            if not isinstance(value, dict):
                continue
            setattr(cls, key, LocalFile(value))


def sjoin(left: Combo, right: Combo) -> tuple[ndarray, ndarray]:
    _ = left.geometry, right.geometry
    left = left.assign(iloc=np.arange(len(left)))
    right = right.assign(iloc=np.arange(len(right)))
    matches = (
        left
        .reset_index()
        ['geometry iloc'.split()]
        .sjoin(right.reset_index()['geometry iloc'.split()])
    )
    ileft = matches.iloc_left.values
    iright = matches.iloc_right.values
    return ileft, iright


def intersection(
        left: boxes.Base,
        right: boxes.Base,
) -> ndarray[float]:
    """area of the intersection"""
    assert len(left) == len(right), f'left and right must have the same length, got {len(left)} and {len(right)}'
    w = np.maximum(left.w.values, right.w.values)
    e = np.minimum(left.e.values, right.e.values)
    s = np.maximum(left.s.values, right.s.values)
    n = np.minimum(left.n.values, right.n.values)
    width = np.maximum(0, e - w)
    height = np.maximum(0, n - s)
    result = width * height
    return result

def union(
        left: boxes.Base,
        right: boxes.Base,
) -> ndarray[float]:
    """area of the union"""
    intersection_area = intersection(left, right)
    area_left = (left.e.values - left.w.values) * (left.n.values - left.s.values)
    area_right = (right.e.values - right.w.values) * (right.n.values - right.s.values)
    union_area = area_left + area_right - intersection_area
    return union_area


def match(
        left: boxes.Base,
        right: boxes.Base,
        threshold: float = .9,
) -> ndarray:
    """
    Using Maryam's suggested approach:

    For each GT box, we find one box across all predictions
    that have the highest overlap with it and then find other
    prediction boxes that have more than 90% overlap with that box
    found in predictions.
    """
    LEFT = left
    RIGHT = right
    _ = left.area, right.area
    ileft, iright = sjoin(left, right)
    left = LEFT.iloc[ileft]
    right = RIGHT.iloc[iright]

    if left is right:
        loc = ileft != iright
        ileft = ileft[loc]
        iright = iright[loc]

    # box in right which has the highest overlap with left
    area = intersection(left, right)
    intersection = area / left.area.values
    ibest = (
        Series(intersection)
        .groupby(ileft)
        .idxmax()
        .loc[ileft]
        .values
    )
    ibest = iright[ibest]
    best = RIGHT.iloc[ibest]

    # boxes in right that have more than 90% overlap with best
    area = intersection(best, right)
    intersection = area / right.area.values
    loc = intersection >= threshold

    ileft = ileft[loc]
    iright = ibest[loc]
    result = np.concatenate([ileft, iright])
    return result

def to_file(self: pd.DataFrame, path: Path, *args, **kwargs):
    path = Path(path)
    try:
        func =  getattr(self, f'to_{path.extension}')
    except AttributeError:
        raise ValueError(f'Extension {path.extension} not supported')
    func(path, *args, **kwargs)


def from_file(cls, path, *args, **kwargs):
    # if isinstance(cls, gpd.DataFrame):
    try:
        return getattr(cls, f'from_{path.suffix[1:]}')
    except AttributeError:
        ...
    else:
        if isinstance(cls, gpd.GeoDataFrame):
            func = getattr(gpd, f'read_{path.suffix[1:]}')
        else:
            func = getattr(pd, f'read_{path.suffix[1:]}')
        frame = func(path, *args, **kwargs)
    return cls(frame)


def compute_average_precision(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """ Compute the Average Precision (AP). """
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))

    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def draw_text_with_outline(draw, xy, text, font, text_color, outline_color, outline_width):
    x, y = xy
    # Draw the outline by drawing the text in the outline color shifted in all directions
    draw.text((x - outline_width, y - outline_width), text, font=font, fill=outline_color)
    draw.text((x + outline_width, y - outline_width), text, font=font, fill=outline_color)
    draw.text((x - outline_width, y + outline_width), text, font=font, fill=outline_color)
    draw.text((x + outline_width, y + outline_width), text, font=font, fill=outline_color)
    # Draw the main text on top
    draw.text((x, y), text, font=font, fill=text_color)


#
# try:
#     func = getattr(cls, f'from_{path.suffix[1:]}')
# except AttributeError:
#     ...
# else:
#     result = func(path)
#     result = cls(result)
#     result.passed = path
#     return result
# try:
#     func = getattr(pd, f'read_{path.suffix[1:]}')
# except AttributeError:
#     raise ValueError(f'Unsupported file type: {path.suffix}')
# else:
#     frame = func(path)
#     result = cls(frame)
#     result.passed = path
#     return result

if __name__ == '__main__':
    ...


    from PIL import Image

