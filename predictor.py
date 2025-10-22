import domain

from dataclasses import dataclass
from collections import defaultdict
from typing import Protocol, Iterator, Optional

import numpy as np
import torch
from torchvision.ops import nms
from shapely.geometry import Polygon
from shapely.ops import unary_union


class WSIIterator(Protocol):
    def iter_first_section_bgr(
        self, window_size: int, overlap_ratio: float
    ) -> Iterator[tuple[np.ndarray, domain.Coords]]: ...


class Model(Protocol):
    def predict(
        self, images: list[np.ndarray]
    ) -> dict[str, list[domain.Prediction]]: ...


@dataclass
class ModelConfig:
    model: Model
    window_size: int


class WSIPredictor:
    def __init__(
        self,
        wsi_iterator: WSIIterator,
        model_configs: list[ModelConfig],
        overlap_ratio: float = 0.5,
    ):
        self.iterator = wsi_iterator
        self.model_configs = model_configs
        self.overlap_ratio = overlap_ratio

    def predict_first_section(self) -> dict[str, list[domain.Prediction]]:
        preds = self._predict_first_section()
        return self._postprocess_predictions(preds)

    def _predict_first_section(self) -> dict[str, list[domain.Prediction]]:
        size_to_models = defaultdict(list[Model])
        for cfg in self.model_configs:
            size_to_models[cfg.window_size].append(cfg.model)

        predictions = defaultdict(list[domain.Prediction])

        for window_size, models in size_to_models.items():
            for region, start in self.iterator.iter_first_section_bgr(
                window_size=window_size,
                overlap_ratio=self.overlap_ratio,
            ):
                for model in models:
                    window_preds = model.predict([region])

                    for cls_name, preds in window_preds.items():
                        for p in preds:
                            predictions[cls_name].append(
                                domain.Prediction(
                                    box=self._to_absolute_box(start, p.box),
                                    polygon=self._to_absolute_polygon(start, p.polygon),
                                    conf=p.conf,
                                )
                            )

        return predictions

    def _postprocess_predictions(
        self, preds: dict[str, list[domain.Prediction]]
    ) -> dict[str, list[domain.Prediction]]:
        filtered_preds = {}

        for pathology, pred in preds.items():
            result = self.__nms(pred, 0.5)
            result = self.__merge_by_coverage(result, 0.5)
            filtered_preds[pathology] = result

        return filtered_preds

    def __nms(self, preds: list[domain.Prediction], iou_threshold: float):
        boxes = torch.tensor(
            [[p.box.start.x, p.box.start.y, p.box.end.x, p.box.end.y] for p in preds]
        )
        scores = torch.tensor([p.conf for p in preds])
        keep_indices = nms(boxes, scores, iou_threshold)

        return [preds[i] for i in keep_indices]

    def __merge_by_coverage(
        self,
        preds: list[domain.Prediction],
        coverage_threshold: float,
    ) -> list[domain.Prediction]:
        if not preds:
            return []

        preds = sorted(preds, key=lambda x: (x.box.area(), x.conf), reverse=True)
        result = []
        for pred in preds:
            merged = pred
            merged_indices = []

            for i, selected_pred in enumerate(result):
                max_coverage = max(
                    pred.box.covered_by(selected_pred.box),
                    selected_pred.box.covered_by(pred.box),
                )

                if max_coverage > coverage_threshold:
                    merged = self._merge_predictions(merged, selected_pred)
                    merged_indices.append(i)

            if merged_indices:
                for i in sorted(merged_indices, reverse=True):
                    result.pop(i)
                result.append(merged)
            else:
                result.append(pred)

        return result

    def _merge_predictions(
        self, pred1: domain.Prediction, pred2: domain.Prediction
    ) -> domain.Prediction:
        merged_box = pred1.box.merge_with(pred2.box)
        merged_conf = max(pred1.conf, pred2.conf)
        merged_polygon = self._merge_polygons(pred1.polygon, pred2.polygon)

        return domain.Prediction(
            box=merged_box,
            conf=merged_conf,
            polygon=merged_polygon,
        )

    def _merge_polygons(
        self,
        poly_a: Optional[domain.Polygon],
        poly_b: Optional[domain.Polygon],
    ) -> Optional[domain.Polygon]:
        if poly_a is None and poly_b is None:
            return None
        if poly_a is None:
            return poly_b
        if poly_b is None:
            return poly_a

        pa = self._to_shapely_polygon(poly_a)
        pb = self._to_shapely_polygon(poly_b)

        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        merged = unary_union([pa, pb])

        if merged.geom_type == "MultiPolygon":
            merged = max(merged.geoms, key=lambda g: g.area)

        return domain.Polygon([domain.Coords(x, y) for x, y in merged.exterior.coords])

    def _to_shapely_polygon(self, polygon: domain.Polygon) -> Polygon:
        return Polygon((p.x, p.y) for p in polygon.coords)

    def _to_absolute_polygon(
        self, start: domain.Coords, poly: Optional[domain.Polygon]
    ) -> Optional[domain.Polygon]:
        if not poly:
            return None

        coords = [domain.Coords(c.x + start.x, c.y + start.y) for c in poly.coords]
        return domain.Polygon(coords=coords)

    def _to_absolute_box(self, start: domain.Coords, box: domain.Box) -> domain.Box:
        return domain.Box(
            start=domain.Coords(box.start.x + start.x, box.start.y + start.y),
            end=domain.Coords(box.end.x + start.x, box.end.y + start.y),
        )
