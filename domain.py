from dataclasses import dataclass
from typing import Optional


@dataclass
class Coords:
    x: float
    y: float


@dataclass
class Polygon:
    coords: list[Coords]


@dataclass
class Box:
    start: Coords
    end: Coords

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


@dataclass(frozen=True)
class Prediction:
    box: Box
    conf: float
    polygon: Optional[Polygon]
