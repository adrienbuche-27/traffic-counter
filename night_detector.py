# night_detector.py — détection de passage par phares (MOG2)
import cv2
import numpy as np
import logging
from collections import defaultdict
import config

log = logging.getLogger("traffic")

class NightDetector:
    """
    Détecte les passages de véhicules la nuit via leurs phares.
    Utilise MOG2 (soustraction de fond) + suivi de centroïde lumineux.
    Pas de classification — retourne uniquement direction et timestamp.
    """

    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESHOLD,
            detectShadows=False
        )
        # Suivi simplifié des blobs lumineux
        # blob_id → liste de positions X
        self.blob_history  = defaultdict(list)
        self.counted_blobs = set()
        self.next_blob_id  = 0
        self.active_blobs  = {}   # blob_id → centroïde courant (cx, cy)

        log.info("NightDetector initialisé (MOG2)")

    def reset(self):
        """Appelé lors du retour en mode jour pour repartir proprement."""
        self.blob_history.clear()
        self.counted_blobs.clear()
        self.active_blobs.clear()
        self.next_blob_id = 0
        log.info("NightDetector réinitialisé")

    def _match_blob(self, cx, cy) -> int:
        """
        Associe un centroïde détecté à un blob existant (distance < 80px)
        ou crée un nouveau blob. Retourne le blob_id.
        """
        for bid, (bx, by) in self.active_blobs.items():
            if abs(cx - bx) < 80 and abs(cy - by) < 60:
                return bid
        # Nouveau blob
        bid = self.next_blob_id
        self.next_blob_id += 1
        return bid

    def process(self, frame: np.ndarray) -> list[dict]:
        """
        Traite une frame et retourne la liste des passages détectés.
        Chaque passage : {"direction": "→" ou "←", "class": "unknown"}
        """
        passages = []

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Amplifier les zones lumineuses (phares)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Appliquer MOG2 sur le masque lumineux
        fg_mask = self.mog2.apply(bright_mask)

        # Morphologie pour nettoyer le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)

        # Trouver les contours des blobs lumineux
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        current_blobs = {}

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MOG2_MIN_AREA:
                continue   # trop petit = bruit

            M  = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            bid = self._match_blob(cx, cy)
            current_blobs[bid] = (cx, cy)

            self.blob_history[bid].append((cx, cy))
            if len(self.blob_history[bid]) > 30:
                self.blob_history[bid].pop(0)

            # ── Détection de franchissement ──
            positions_x = [p[0] for p in self.blob_history[bid]]

            if bid not in self.counted_blobs and len(positions_x) >= 3:
                prev_x = positions_x[-2]
                curr_x = positions_x[-1]
                crossed = (
                    (prev_x < config.COUNT_LINE_X <= curr_x) or
                    (prev_x > config.COUNT_LINE_X >= curr_x)
                )
                if crossed:
                    direction = "→" if curr_x > prev_x else "←"
                    log.info(
                        f"[NUIT] Passage phares détecté  blob#{bid}  {direction}  "
                        f"prev_x={prev_x}  curr_x={curr_x}  ligne={config.COUNT_LINE_X}"
                    )
                    passages.append({"direction": direction, "class": "unknown"})
                    self.counted_blobs.add(bid)

        # Nettoyage des blobs disparus
        for bid in list(self.active_blobs.keys()):
            if bid not in current_blobs:
                del self.active_blobs[bid]
                self.blob_history.pop(bid, None)

        self.active_blobs = current_blobs
        return passages