from dataclasses import dataclass
from typing import Optional
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
            return Polygon(list(largest.exterior.coords))

        return Polygon(list(new_polygon.exterior.coords))  # type: ignore

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
