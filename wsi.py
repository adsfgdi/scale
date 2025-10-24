import domain

import cv2
import numpy as np
from typing import Iterator, Optional, Tuple

from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cucim.clara import CuImage


class WSI:
    def __init__(
        self,
        wsi_path: str,
        num_sections: Optional[int] = None,
    ):
        self.num_sections = num_sections
        self.min_thumb_width = 1024

        self.wsi = CuImage(wsi_path)
        self.wsi_size = self.wsi.resolutions["level_dimensions"][0]

        self.thumb = self._create_wsi_thumbnail()
        self.thumb_size = (self.thumb.shape[1], self.thumb.shape[0])

    def iter_first_section_bgr(
        self, window_size: int, overlap_ratio: float
    ) -> Iterator[Tuple[np.ndarray, domain.Coords]]:
        assert 0 <= overlap_ratio < 1, "overlap_ratio должен быть в диапазоне [0, 1)"

        first_bound = self.extract_first_biopsy_bound()

        x_start, y_start = int(first_bound.start.x), int(first_bound.start.y)
        w, h = map(int, first_bound.size())

        stride = int(window_size * (1 - overlap_ratio))

        for window_y in range(0, h, stride):
            for window_x in range(0, w, stride):
                size_x = min(window_size, w - window_x)
                size_y = min(window_size, h - window_y)

                region = self.wsi.read_region(
                    location=(x_start + window_x, y_start + window_y),
                    size=(size_x, size_y),
                    level=0,
                )
                region = cv2.cvtColor(np.asarray(region)[..., :3], cv2.COLOR_RGB2BGR)

                # Координаты верхнего левого угла текущего окна в глобальной системе координат WSI.
                start = domain.Coords((x_start + window_x), (y_start + window_y))

                yield region, start

    def visualize_wsi_sections(self) -> np.ndarray:
        bounds = self.extract_biopsy_bounds()

        img = self.thumb.copy()

        scale_x = self.thumb_size[0] / self.wsi_size[0]
        scale_y = self.thumb_size[1] / self.wsi_size[1]

        for box in bounds:
            x1 = int(box.start.x * scale_x)
            y1 = int(box.start.y * scale_y)
            x2 = int(box.end.x * scale_x)
            y2 = int(box.end.y * scale_y)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        return img

    def extract_first_biopsy_section(self) -> np.ndarray:
        return self.extract_biopsy_section(0)

    def extract_biopsy_section(self, index: int) -> np.ndarray:
        bound = self.extract_biopsy_bound(index)

        crop = self.wsi.read_region(
            location=(bound.start.x, bound.start.y),
            size=bound.size(),
            level=0,
        )

        return np.asarray(crop)[..., :3]

    def extract_biopsy_sections(self) -> list[np.ndarray]:
        bounds = self.extract_biopsy_bounds()

        sections = []
        for bound in bounds:
            crop = self.wsi.read_region(
                location=(bound.start.x, bound.start.y), size=bound.size(), level=0
            )

            crop_rgb = np.asarray(crop)[..., :3]
            sections.append(crop_rgb)

        return sections

    def extract_first_biopsy_bound(self) -> domain.Box:
        return self.extract_biopsy_bound(0)

    def extract_biopsy_bound(self, index: int) -> domain.Box:
        return self.extract_biopsy_bounds()[index]

    def extract_biopsy_bounds(self) -> list[domain.Box]:
        clusters = self._cluster_biopsies(self.thumb)

        return self._extract_biopsy_bounds(clusters, self.thumb_size)

    def _extract_biopsy_bounds(
        self, biopsy_clusters, thumbnail_size
    ) -> list[domain.Box]:
        biopsy_regions = []

        for cluster in biopsy_clusters:
            min_row = min(r.bbox[0] for r in cluster)
            min_col = min(r.bbox[1] for r in cluster)
            max_row = max(r.bbox[2] for r in cluster)
            max_col = max(r.bbox[3] for r in cluster)

            scale_x = self.wsi_size[0] / thumbnail_size[0]
            scale_y = self.wsi_size[1] / thumbnail_size[1]

            box = domain.Box(
                domain.Coords(int(min_col * scale_x), int(min_row * scale_y)),
                domain.Coords(int(max_col * scale_x), int(max_row * scale_y)),
            )

            biopsy_regions.append(box)

        return sorted(biopsy_regions, key=lambda x: (x.start.y, x.start.x))

    def _detect_tissue_regions(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_tissue = np.array([0, 30, 30])
        upper_tissue = np.array([180, 255, 255])
        tissue_mask = cv2.inRange(hsv, lower_tissue, upper_tissue)

        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)

        labeled_image = label(tissue_mask)
        regions = regionprops(labeled_image)

        areas = np.array([r.area for r in regions])
        min_area = np.median(areas) * 0.3

        return [r for r in regions if r.area >= min_area]

    def _cluster_biopsies(self, img):
        regions = self._detect_tissue_regions(img)

        if not regions:
            return []

        centroids = np.array([[r.centroid[1], r.centroid[0]] for r in regions])

        if self.num_sections:
            n_clusters = min(self.num_sections, len(regions))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(centroids)
        else:
            labels = self._find_optimal_clusters(centroids)

        n_clusters = len(set(labels))
        clusters = [[] for _ in range(n_clusters)]
        for region, label in zip(regions, labels):
            clusters[label].append(region)

        return clusters

    def _find_optimal_clusters(self, centroids):
        n_min = 1
        n_max = n_max = min(16, len(centroids) - 1)

        best_score = -1
        best_labels = None

        for n in range(n_min, n_max + 1):
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(centroids)

            if len(set(labels)) < 2:
                continue

            score = silhouette_score(centroids, labels)
            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            kmeans = KMeans(n_clusters=n_min, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(centroids)

        return best_labels

    def _create_wsi_thumbnail(self):
        target_level = self._choose_best_level()
        width, height = self.wsi.resolutions["level_dimensions"][target_level]

        thumbnail = self.wsi.read_region(
            location=(0, 0),
            size=(width, height),
            level=target_level,
        )

        return np.asarray(thumbnail)

    def _choose_best_level(self) -> int:
        widths = [res[0] for res in self.wsi.resolutions["level_dimensions"]]
        candidates = [
            (level, w) for level, w in enumerate(widths) if w >= self.min_thumb_width
        ]

        if candidates:
            return min(candidates, key=lambda x: x[1])[0]

        return int(np.argmax(widths))
