from . import domain

from typing import Protocol, Optional
import colorsys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon

matplotlib.use("Agg")


class Extractor(Protocol):
    def extract_first_biopsy_bound(self) -> domain.Box: ...
    def extract_first_biopsy_section(self) -> np.ndarray: ...


class Visualizer:
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def visualize_first_section_with_predictions(
        self,
        predictions: dict[str, list[domain.Prediction]],
        target_class: Optional[str] = None,
    ) -> np.ndarray:
        section = self.extractor.extract_first_biopsy_section()
        first_bound = self.extractor.extract_first_biopsy_bound()

        if target_class is not None:
            predictions = {
                cls: preds for cls, preds in predictions.items() if cls == target_class
            }

        local_preds = self._to_local_predictions(predictions, first_bound.start)
        return self._visualize_predictions(section, local_preds)

    def _visualize_predictions(
        self,
        img_bgr: np.ndarray,
        predictions: dict[str, list[domain.Prediction]],
    ) -> np.ndarray:
        # BGR → RGB для matplotlib
        img_rgb = img_bgr[..., ::-1] if img_bgr.ndim == 3 else img_bgr

        height, width = img_rgb.shape[:2]
        dpi = 100
        fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        ax.imshow(img_rgb)
        ax.set_aspect("equal")

        color_map = self._color_map(predictions)

        for cls_name, preds in predictions.items():
            color = self._normalize_color(color_map.get(cls_name, (0, 255, 0)))
            for p in preds:
                self._draw_box(p, color, ax)
                if p.polygon:
                    self._draw_polygon(p, color, ax)

        self._draw_legend(ax, color_map)

        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.canvas.draw()
        img_rgba = np.asarray(fig.canvas.buffer_rgba())
        img_bgr_result = img_rgba[..., [2, 1, 0]]  # RGBA → BGR

        plt.close(fig)
        return img_bgr_result

    def _draw_box(
        self, pred: domain.Prediction, color: tuple[float, float, float], ax: plt.Axes
    ):
        x1, y1 = pred.box.start.x, pred.box.start.y
        width = pred.box.end.x - x1
        height = pred.box.end.y - y1
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=3,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    def _draw_polygon(
        self, pred: domain.Prediction, color: tuple[float, float, float], ax: plt.Axes
    ):
        if not pred.polygon:
            return
        pts = np.array([[p.x, p.y] for p in pred.polygon.to_coords()])
        polygon = MplPolygon(
            pts, closed=True, facecolor=(*color, 0.1), edgecolor=color, linewidth=2
        )
        ax.add_patch(polygon)

    def _draw_legend(self, ax: plt.Axes, color_map: dict[str, tuple[int, int, int]]):
        if not color_map:
            return

        bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
        fig_width_inch = bbox.width
        base_font = fig_width_inch * 0.8

        legend_patches = [
            patches.Patch(facecolor=self._normalize_color(c), label=cls)
            for cls, c in color_map.items()
        ]

        leg = ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            framealpha=0.8,
            fontsize=base_font,
            edgecolor="none",
            labelcolor="white",
        )

        frame = leg.get_frame()
        frame.set_facecolor((0, 0, 0, 0.6))
        frame.set_edgecolor("none")

        for text in leg.get_texts():
            text.set_color("white")

    def _color_map(
        self, predictions: dict[str, list[domain.Prediction]]
    ) -> dict[str, tuple[int, int, int]]:
        classes = sorted(predictions.keys())
        n = max(len(classes), 1)
        color_map = {}
        for i, cls in enumerate(classes):
            hue = i / n
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
            color_map[cls] = (int(r * 255), int(g * 255), int(b * 255))
        return color_map

    @staticmethod
    def _normalize_color(color: tuple[int, int, int]) -> tuple[float, float, float]:
        return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    def _to_local_predictions(
        self,
        preds_by_class: dict[str, list[domain.Prediction]],
        origin: domain.Coords,
    ) -> dict[str, list[domain.Prediction]]:
        return {
            cls: [
                domain.Prediction(
                    box=self._shift_box(p.box, origin),
                    polygon=self._shift_polygon(p.polygon, origin),
                    conf=p.conf,
                )
                for p in preds
            ]
            for cls, preds in preds_by_class.items()
        }

    @staticmethod
    def _shift_box(box: domain.Box, origin: domain.Coords) -> domain.Box:
        return domain.Box(
            start=domain.Coords(box.start.x - origin.x, box.start.y - origin.y),
            end=domain.Coords(box.end.x - origin.x, box.end.y - origin.y),
        )

    @staticmethod
    def _shift_polygon(
        polygon: Optional[domain.Polygon], origin: domain.Coords
    ) -> Optional[domain.Polygon]:
        if polygon is None:
            return None

        return domain.Polygon(
            [domain.Coords(p.x - origin.x, p.y - origin.y) for p in polygon.to_coords()]
        )
