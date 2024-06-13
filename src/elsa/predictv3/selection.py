from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from dataclasses import dataclass
from functools import *
from matplotlib.colors import to_rgb
from numpy import ndarray
from typing import *

import elsa.util as util


@dataclass
class Selection:
    prompt: ndarray
    xywh: ndarray
    label: ndarray
    labels: ndarray
    box_threshold: float
    file: str
    natural: str

    @cached_property
    def boolean_mask(self):
        result = self.prompt.max(axis=1) > self.box_threshold
        return result

    @cached_property
    def norm(self):
        result = (
            self.label.max
            (axis=0)
            [None, :]
            .__rtruediv__(self.label)
        )
        return result

    def view_distribution(
            self,
            label: str,
            color: str = 'white',
            facecolor: str = 'black',
    ) -> plt.Figure:
        for i, l in enumerate(self.labels):
            if l == label:
                break
        else:
            raise ValueError(f'Label {label} not found')

        confidence: np.ndarray = self.label[:, i]
        sorted_confidence = np.sort(confidence)

        fig, ax = plt.subplots()
        ax.plot(sorted_confidence, color=color)
        ax.set_facecolor(facecolor)
        fig.patch.set_facecolor(facecolor)
        ax.spines['bottom'].set_color(color)
        ax.spines['top'].set_color(color)
        ax.spines['left'].set_color(color)
        ax.spines['right'].set_color(color)
        ax.tick_params(axis='x', colors=color)
        ax.tick_params(axis='y', colors=color)
        ax.set_ylabel('confidence', color=color)
        ax.set_xlabel('sorted rank/900', color=color)
        ax.yaxis.label.set_color(color)
        ax.xaxis.label.set_color(color)

        file = (
            str(self.file)
            .rsplit(os.sep, 1)[-1]
            .rsplit('.', 1)[0]
        )
        title = f"'{label}' from '{self.natural}'"
        ax.set_title(title, color=color)

        return fig

    def view_stacked_distributions(
            self,
            threshold: float = None,
    ) -> plt.Figure:
        num_labels = len(self.labels)
        fig, axs = plt.subplots(num_labels, 1, figsize=(8, num_labels * 2))
        if num_labels == 1:
            axs = [axs]  # Ensure axs is always iterable

        if threshold is None:
            loc = (
                self.label
                .__gt__(self.label.mean(axis=0))
                .any(axis=1)
            )
        else:
            loc = (
                    self.label
                    .max(axis=1)
                    >= threshold
            )

        for ax, label in zip(axs, self.labels):
            i = 0
            for i, l in enumerate(self.labels):
                if l == label:
                    break

            confidence: np.ndarray = self.label[loc, i]
            indices = np.where(loc)[0]

            ax.scatter(indices, confidence, color='white', s=10)  # Use scatter plot for dots
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_ylabel('confidence', color='white')
            ax.set_xlabel('index', color='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.set_xlim(0, 900)  # Ensure the x-axis covers the whole set from 0 to 900

            title = f"'{label}' from '{self.natural}'"
            ax.set_title(title, color='white')

        fig.tight_layout()
        fig.patch.set_facecolor('black')

        return fig

    import matplotlib.pyplot as plt

    def view_distributions(
            self,
            color: str = 'white',
            facecolor: str = 'black',
    ) -> Iterator[plt.Figure]:
        yield from (
            self.view_distribution(l, color, facecolor)
            for l in self.labels
        )

    def view_heatmap(
            self,
            label: str,
            color: str = 'red',
            norm: bool = False,
    ) -> Image:
        image = Image.open(self.file).convert("RGBA")
        rgba = *to_rgb(color), 0
        shape = *image.size[::-1], 4
        overlay = np.full(shape, rgba, dtype=np.uint8) * 255
        width, height = image.size

        for i, l in enumerate(self.labels):
            if l == label:
                break
        else:
            raise ValueError(f'Label {label} not found')

        if norm:
            confidence: np.ndarray = self.norm[:, i]
        else:
            confidence: np.ndarray = self.label[:, i]
        xywh = self.xywh.copy()
        xywh[:, [0, 2]] *= width
        xywh[:, [1, 3]] *= height
        ws = xywh[:, :2] - xywh[:, 2:] / 2
        en = xywh[:, :2] + xywh[:, 2:] / 2
        assert (en >= ws).all()
        ws = ws.astype(int)
        en = en.astype(int)
        iloc = confidence.argsort()
        alpha = confidence[iloc] * 255
        ws = ws[iloc]
        en = en[iloc]
        assert (en >= ws).all()
        it = zip(ws[:, 0], ws[:, 1], en[:, 0], en[:, 1], alpha)
        for w, s, e, n, a in it:
            overlay[s:n, w:e, 3] = a

        overlay_img = Image.fromarray(overlay, mode="RGBA")
        combined_img = Image.alpha_composite(image, overlay_img)
        print(f'Label: {label}')
        return combined_img

    def view_heatmaps(
            self,
            color: str = 'blue',
    ) -> Iterator[plt.Figure]:
        yield from (
            self.view_heatmap(l, color)
            for l in self.labels
        )

    def view_stacked_heatmaps(
            self,
            color: str = 'blue',
    ) -> plt.Figure:
        """ opacity is confidence / confidence.max() """
        num_labels = len(self.labels)
        fig, axs = plt.subplots(num_labels, 1, figsize=(10, num_labels * 6))
        if num_labels == 1:
            axs = [axs]  # Ensure axs is always iterable

        image = Image.open(self.file).convert("RGBA")
        width, height = image.size

        for ax, label in zip(axs, self.labels):
            i = 0
            for i, l in enumerate(self.labels):
                if l == label:
                    break

            # opacity is normalized value
            confidence = self.norm[:, i]
            xywh = self.xywh.copy()
            xywh[:, [0, 2]] *= width
            xywh[:, [1, 3]] *= height
            ws = xywh[:, :2] - xywh[:, 2:] / 2
            en = xywh[:, :2] + xywh[:, 2:] / 2
            assert (en >= ws).all()
            ws = ws.astype(int)
            en = en.astype(int)
            iloc = confidence.argsort()
            alpha = confidence[iloc] * 255
            ws = ws[iloc]
            en = en[iloc]
            assert (en >= ws).all()
            it = zip(ws[:, 0], ws[:, 1], en[:, 0], en[:, 1], alpha)

            rgba = *to_rgb(color), 0
            shape = *image.size[::-1], 4
            overlay = np.full(shape, rgba, dtype=np.uint8) * 255

            for w, s, e, n, a in it:
                overlay[s:n, w:e, 3] = a

            overlay_img = Image.fromarray(overlay, mode="RGBA")
            combined_img = Image.alpha_composite(image, overlay_img)
            max = round(self.label[:, i].max(), 4)

            ax.imshow(combined_img)
            ax.axis('off')
            title = f'{label}; {max=}'
            ax.set_title(title, color='white')

        file = util.trim_path(self.file)
        fig.suptitle(file, color='white', fontsize=16)
        fig.tight_layout()
        fig.patch.set_facecolor('black')

        return fig

    def view_overlaid_heatmaps(
            self,
            red: str | int = 0,
            green: str | int = 1,
            blue: str | int = 2,
    ) -> Image:
        # todo: also get meta
        """
        View the overlaid heatmaps;

        let normLogit = logit / logit.max()
        color is normLogit / sum(normLogit) for each color
        opacity is sum(normLogit) / sum(normLogit).max()
        """
        image = Image.open(self.file).convert("RGBA")
        shape = *image.size[::-1], 4
        nlabels = len(self.labels)
        rows = len(self.label)
        width, height = image.size
        labels = (red, green, blue)
        colors = 'red green blue'.split()
        n = 0
        nactual = sum(bool(c is not None) for c in labels)
        confidences = np.zeros((rows, 3))
        icolors: list[int] = []

        it = zip(labels, colors, range(3))
        for label, color, c in it:
            if label is None:
                continue
            n += 1
            if n > len(self.labels):
                break
            if isinstance(label, str):
                for i, l in enumerate(self.labels):
                    if l == label:
                        label = i
                        break
                else:
                    raise ValueError(f'Label {label} not found')
            elif isinstance(label, int):
                i = label
                label = self.labels[i]
            else:
                raise ValueError(f'Label {label} not understood')
            icolors.append(i)
            confidences[:, c] = self.norm[:, i]

        # total is sum of normalized confidences for each row
        total = confidences.sum(axis=1)
        # contribution is percent of total confidence for each label
        contribution = confidences / total[:, None]
        # relative confidence is row's confidence out of max row confidence
        relative_confidence = total / total.max()

        # sort by confidence
        iloc = relative_confidence.argsort()
        alpha = relative_confidence[iloc] * 255
        rgb = contribution[iloc] * 255
        overlay = np.full(shape, 0, dtype=np.uint8)

        # convert normalized xywh to absolute wsen
        xywh = self.xywh.copy()
        xywh[:, [0, 2]] *= width
        xywh[:, [1, 3]] *= height
        ws = xywh[:, :2] - xywh[:, 2:] / 2
        en = xywh[:, :2] + xywh[:, 2:] / 2
        assert (en >= ws).all()
        ws = ws.astype(int)
        en = en.astype(int)
        ws = ws[iloc]
        en = en[iloc]
        it = zip(ws[:, 0], ws[:, 1], en[:, 0], en[:, 1], alpha, rgb)
        for w, s, e, n, a, c in it:
            overlay[s:n, w:e, 3] = a
            overlay[s:n, w:e, :3] = c

        overlay_img = Image.fromarray(overlay, mode="RGBA")
        combined_img = Image.alpha_composite(image, overlay_img)

        draw = ImageDraw.Draw(combined_img)
        font = ImageFont.load_default()
        it = zip(labels, colors, icolors)
        for label, color, icolor in it:
            if isinstance(label, int):
                label_name = self.labels[label]
            else:
                label_name = label
            max = round(self.label[:, icolor].max(), 4)
            title = f'{label_name}; {max=}'
            draw.text((10, 40 + 30 * colors.index(color)), title, fill=color, font=font)

        file = util.trim_path(self.file)
        draw.text((10, 10), file, fill='white', font=font)

        return combined_img
