from . import domain

from dataclasses import dataclass
from collections import defaultdict
from typing import Protocol, Iterator, Optional
import time
from contextlib import contextmanager

import numpy as np
import torch
from torchvision.ops import nms


@contextmanager
def timer(name: str, enabled: bool = True):
    start = time.perf_counter()
    yield
    if enabled:
        elapsed = time.perf_counter() - start
        print(f"[TIMER] {name}: {elapsed:.3f} s")


class WSIIterator(Protocol):
    def iter_section_bgr(
        self, index: int, window_size: int, overlap_ratio: float
    ) -> Iterator[tuple[np.ndarray, domain.Coords]]: ...


class Model(Protocol):
    def predict(
        self, images: list[np.ndarray]
    ) -> dict[str, list[domain.Prediction]]: ...


@dataclass
class Postprocessing:
    nms_thres: Optional[float]
    cov_thres: Optional[float]


@dataclass
class ModelConfig:
    model: Model
    window_size: int


class WSIPredictor:
    def __init__(
        self,
        wsi_iterator: WSIIterator,
        model_configs: list[ModelConfig],
        postprocess_settings: dict[str, Postprocessing],
        overlap_ratio: float = 0.5,
        enable_timing: bool = False,
    ):
        self.iterator = wsi_iterator
        self.model_configs = model_configs
        self.overlap_ratio = overlap_ratio
        self.postprocess_settings = postprocess_settings
        self.enable_timing = enable_timing

    def predict_section(self, section_index: int) -> dict[str, list[domain.Prediction]]:
        with timer(f"_predict_section(index={section_index})", self.enable_timing):
            preds_all = self._predict_section(sec_index=section_index)
        return self._postprocess(preds_all)

    def predict_first_section(self) -> dict[str, list[domain.Prediction]]:
        with timer("_predict_section(index=0)", self.enable_timing):
            preds_all = self._predict_section(sec_index=0)
        return self._postprocess(preds_all)

    def _postprocess(
        self, preds_all: dict[str, list[domain.Prediction]]
    ) -> dict[str, list[domain.Prediction]]:
        result = {}
        for cls_name, preds in preds_all.items():
            with timer(f"_postprocess_predictions({cls_name})", self.enable_timing):
                result[cls_name] = self._postprocess_predictions(preds, cls_name)

        return result

    def _predict_section(self, sec_index: int) -> dict[str, list[domain.Prediction]]:
        size_to_models = self._group_models_by_window_size()
        predictions = defaultdict(list[domain.Prediction])

        for window_size, models in size_to_models.items():
            for region, start in self.iterator.iter_section_bgr(
                index=sec_index,
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
        self, preds: list[domain.Prediction], cls_name: str
    ) -> list[domain.Prediction]:
        settings = self.postprocess_settings[cls_name]
        if settings is None:
            raise ValueError(f"No postprocessing settings found for class '{cls_name}'")

        if settings.nms_thres:
            preds = self.__bbox_nms(preds, settings.nms_thres)
        if settings.cov_thres:
            preds = self.__merge_by_coverage(preds, settings.cov_thres)

        return preds

    def __bbox_nms(self, preds: list[domain.Prediction], iou_threshold: float):
        if not preds:
            return []

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
        result: list[domain.Prediction] = []
        for pred in preds:
            merged = pred
            merged_indices = []

            for i, selected_pred in enumerate(result):
                max_coverage = max(
                    pred.covered_by(selected_pred), selected_pred.covered_by(pred)
                )

                if max_coverage > coverage_threshold:
                    merged = merged.merge_with(selected_pred)
                    merged_indices.append(i)

            if merged_indices:
                for i in sorted(merged_indices, reverse=True):
                    result.pop(i)
                result.append(merged)
            else:
                result.append(pred)

        return result

    def _group_models_by_window_size(self) -> dict[int, list[Model]]:
        size_to_models = defaultdict(list[Model])
        for cfg in self.model_configs:
            size_to_models[cfg.window_size].append(cfg.model)
        return size_to_models

    def _to_absolute_polygon(
        self, start: domain.Coords, poly: Optional[domain.Polygon]
    ) -> Optional[domain.Polygon]:
        if not poly:
            return None

        coords = [domain.Coords(c.x + start.x, c.y + start.y) for c in poly.to_coords()]
        return domain.Polygon(coords=coords)

    def _to_absolute_box(self, start: domain.Coords, box: domain.Box) -> domain.Box:
        return domain.Box(
            start=domain.Coords(box.start.x + start.x, box.start.y + start.y),
            end=domain.Coords(box.end.x + start.x, box.end.y + start.y),
        )
