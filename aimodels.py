import domain

import numpy as np
from collections import defaultdict

from ultralytics import YOLO


class YOLOModel:
    def __init__(
        self,
        model_path: str,
        min_conf: float,
        cls_index_to_name: dict[int, str],
    ):
        self.min_conf = min_conf
        self.model = YOLO(model_path)
        self.cls_index_to_name = cls_index_to_name

    def predict(self, images: list[np.ndarray]) -> dict[str, list[domain.Prediction]]:
        predictions = self.model.predict(images, verbose=False, conf=self.min_conf)
        return self._parse_yolo_predictions(predictions)

    def _parse_yolo_predictions(
        self, preds: list
    ) -> dict[str, list[domain.Prediction]]:
        parsed_preds = defaultdict(list)

        for pred in preds:
            boxes = pred.boxes.xyxy.tolist()
            confs = pred.boxes.conf.tolist()
            cls_indexes = pred.boxes.cls.tolist()

            has_masks = hasattr(pred, "masks") and pred.masks is not None

            polygons = []
            if has_masks:
                polygons = pred.masks.xy

            for i, (box, cls_index, conf) in enumerate(zip(boxes, cls_indexes, confs)):
                if conf < self.min_conf:
                    continue

                x1, y1, x2, y2 = box
                pathology_type = self.cls_index_to_name[cls_index]

                polygon = None
                if has_masks:
                    coords = [domain.Coords(float(x), float(y)) for x, y in polygons[i]]
                    polygon = domain.Polygon(coords=coords)

                parsed_box = domain.Prediction(
                    domain.Box(start=domain.Coords(x1, y1), end=domain.Coords(x2, y2)),
                    conf=conf,
                    polygon=polygon,
                )
                parsed_preds[pathology_type].append(parsed_box)

        return parsed_preds
