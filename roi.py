# roi.py — masque polygonal + crop pour YOLO
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Optional

ROI_PATH = Path(__file__).parent / "data" / "roi.json"

class ROI:
    """
    Masque polygonal à 4 points.
    Applique le masque et crop sur la bounding box du polygone.
    YOLO reçoit une image réduite → inférence plus rapide.
    Les coordonnées des détections sont remises dans le référentiel original.
    """

    def __init__(self, roi_path: Path = ROI_PATH):
        self.enabled  = False
        self.points   = None
        self.mask     = None
        self.bbox     = None   # (x, y, w, h) de la bounding box du polygone
        self._load(roi_path)

    def _load(self, roi_path: Path):
        if not roi_path.exists():
            return
        with open(roi_path) as f:
            data = json.load(f)
        self.points  = np.array(data["points"], dtype=np.int32)
        self.enabled = True

        # Bounding box du polygone
        x, y, w, h    = cv2.boundingRect(self.points)
        self.bbox      = (x, y, w, h)

        # Masque pleine résolution
        img_w = data["img_width"]
        img_h = data["img_height"]
        self.mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(self.mask, [self.points], 255)

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, tuple]:
        """
        Applique le masque et retourne :
        - frame_cropped : image réduite à envoyer à YOLO
        - offset        : (x_offset, y_offset) pour remettre les coords en place
        """
        if not self.enabled or self.mask is None:
            h, w = frame.shape[:2]
            return frame, (0, 0)

        # Resize le masque si la résolution a changé
        h, w = frame.shape[:2]
        if self.mask.shape != (h, w):
            self.mask = cv2.resize(self.mask, (w, h))
            x, y, bw, bh = self.bbox
            sx = w  / self.points[:, 0].max()
            sy = h  / self.points[:, 1].max()
            scaled = (self.points * [sx, sy]).astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(scaled)
            self.bbox = (x, y, bw, bh)

        # Masque noir hors polygone
        masked = cv2.bitwise_and(frame, frame, mask=self.mask)

        # Crop sur la bounding box
        x, y, bw, bh = self.bbox
        x  = max(0, x)
        y  = max(0, y)
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        cropped = masked[y:y+bh, x:x+bw]
        return cropped, (x, y)

    def restore_coords(self, x1, y1, x2, y2, offset: tuple) -> tuple:
        """Remet les coordonnées YOLO dans le référentiel original."""
        ox, oy = offset
        return x1 + ox, y1 + oy, x2 + ox, y2 + oy

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Dessine la ROI sur une frame pour visualisation."""
        if not self.enabled:
            return frame
        overlay = frame.copy()
        cv2.polylines(frame, [self.points], True, (0, 255, 0), 2)
        cv2.fillPoly(overlay, [self.points], (0, 255, 0))
        return cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)