from dataclasses import dataclass
from typing import Optional
import json
from shapely.geometry import Polygon as ShapelyPolygon


@dataclass
class Coords:
    x: float
    y: float


class Polygon:
    def __init__(self, coords: list[Coords]):
        if len(coords) < 3:
            raise ValueError("Polygon must have at least 3 points")

        polygon = ShapelyPolygon([(c.x, c.y) for c in coords])

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if polygon.geom_type == "MultiPolygon":
            largest = max(polygon.geoms, key=lambda g: g.area)  # type: ignore
            polygon = ShapelyPolygon(list(largest.exterior.coords))

        self._polygon = polygon

    def area(self) -> float:
        return self._polygon.area

    def iou_with(self, other: "Polygon") -> float:
        inter_area = self._intersection_area(other)
        union_area = self._area() + other._area() - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def covered_by(self, other: "Polygon") -> float:
        inter_area = self._intersection_area(other)

        if inter_area == 0:
            return 0.0

        return inter_area / self._area() if self._area() > 0 else 0.0

    def merge_with(self, other: "Polygon") -> "Polygon":
        new_polygon = self._polygon.union(other._polygon)

        if new_polygon.geom_type == "MultiPolygon":
            largest = max(new_polygon.geoms, key=lambda g: g.area)  # type: ignore
            coords = [Coords(x, y) for x, y in largest.exterior.coords]
            return Polygon(coords)

        coords = [Coords(x, y) for x, y in new_polygon.exterior.coords]  # type: ignore
        return Polygon(coords)

    def to_coords(self) -> list[Coords]:
        return list(Coords(c[0], c[1]) for c in self._polygon.exterior.coords)

    def _area(self) -> float:
        return self._polygon.area

    def _intersection_area(self, other: "Polygon") -> float:
        return self._polygon.intersection(other._polygon).area


class Box:
    def __init__(self, start: Coords, end: Coords):
        self.start = start
        self.end = end

    def size(self) -> tuple[float, float]:
        width = self.end.x - self.start.x
        height = self.end.y - self.start.y
        return width, height

    def area(self) -> float:
        width = self.end.x - self.start.x
        height = self.end.y - self.start.y
        return width * height

    def center(self) -> Coords:
        return Coords(
            x=(self.start.x + self.end.x) / 2,
            y=(self.start.y + self.end.y) / 2,
        )

    def intersection_area_with(self, box: "Box") -> float:
        inter_x1 = max(self.start.x, box.start.x)
        inter_y1 = max(self.start.y, box.start.y)
        inter_x2 = min(self.end.x, box.end.x)
        inter_y2 = min(self.end.y, box.end.y)
        return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    def intersects_with(self, other: "Box") -> bool:
        return self.intersection_area_with(other) > 0

    def iou_with(self, other: "Box") -> float:
        inter_area = self.intersection_area_with(other)

        if inter_area == 0:
            return 0.0

        union_area = self.area() + other.area() - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def covered_by(self, other: "Box") -> float:
        inter_area = self.intersection_area_with(other)

        if inter_area == 0:
            return 0.0

        return inter_area / self.area() if self.area() > 0 else 0.0

    def merge_with(self, other: "Box") -> "Box":
        start = Coords(
            x=min(self.start.x, other.start.x),
            y=min(self.start.y, other.start.y),
        )
        end = Coords(
            x=max(self.end.x, other.end.x),
            y=max(self.end.y, other.end.y),
        )
        return Box(start=start, end=end)


class Prediction:
    def __init__(self, box: Box, polygon: Optional[Polygon], conf: float):
        self.box = box
        self.polygon = polygon
        self.conf = conf

    def iou_with(self, other: "Prediction") -> float:
        if self.polygon and other.polygon:
            return self.polygon.iou_with(other.polygon)
        return self.box.iou_with(other.box)

    def covered_by(self, other: "Prediction") -> float:
        if self.polygon and other.polygon:
            return self.polygon.covered_by(other.polygon)
        return self.box.covered_by(other.box)

    def merge_with(self, other: "Prediction") -> "Prediction":
        merged_box = self.box.merge_with(other.box)
        merged_conf = max(self.conf, other.conf)

        merged_poly = None
        if self.polygon and other.polygon:
            merged_poly = self.polygon.merge_with(other.polygon)
        else:
            merged_poly = self.polygon or other.polygon

        return Prediction(box=merged_box, conf=merged_conf, polygon=merged_poly)


def predictions_to_dict(predictions_by_class: dict[str, list[Prediction]]) -> dict:
    data = {}

    for cls_name, preds in predictions_by_class.items():
        data[cls_name] = []
        for p in preds:
            pred_dict = {
                "box": {
                    "start": {"x": p.box.start.x, "y": p.box.start.y},
                    "end": {"x": p.box.end.x, "y": p.box.end.y},
                },
                "polygon": (
                    [{"x": c.x, "y": c.y} for c in p.polygon.to_coords()]
                    if p.polygon
                    else None
                ),
                "conf": p.conf,
            }
            data[cls_name].append(pred_dict)
    return data


def predictions_from_dict(data: dict) -> dict[str, list[Prediction]]:
    predictions_by_class: dict[str, list[Prediction]] = {}

    for cls_name, preds in data.items():
        predictions_by_class[cls_name] = []

        for pdict in preds:
            box_data = pdict["box"]
            start = Coords(**box_data["start"])
            end = Coords(**box_data["end"])
            box = Box(start=start, end=end)

            polygon_data = pdict.get("polygon")
            polygon = None
            if polygon_data:
                polygon = Polygon([Coords(**c) for c in polygon_data])

            conf = pdict["conf"]
            predictions_by_class[cls_name].append(
                Prediction(box=box, polygon=polygon, conf=conf)
            )

    return predictions_by_class


def predictions_to_json(
    predictions_by_class: dict[str, list[Prediction]], path: str
) -> None:
    data = predictions_to_dict(predictions_by_class)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def predictions_from_json(path: str) -> dict[str, list[Prediction]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return predictions_from_dict(data)
