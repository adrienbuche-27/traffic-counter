import cv2

# ══════════════════════════════════════════════════════════════════════════════
# config.py — source unique de vérité pour tous les scripts
# ══════════════════════════════════════════════════════════════════════════════
YOLO_IMGSZ = 320

# ── Caméra — résolution NATIVE du capteur (avant rotation) ────────────────────
# C'est ce que Picamera2 reçoit. Ne jamais changer ces valeurs après rotation.
CAM_NATIVE_WIDTH  = 400
CAM_NATIVE_HEIGHT = 600
CAM_FPS           = 10

# ── Rotation ──────────────────────────────────────────────────────────────────
# Appliquée sur chaque frame après capture, avant tout traitement.
# Options :
#   cv2.ROTATE_90_CLOCKWISE
#   cv2.ROTATE_90_COUNTERCLOCKWISE
#   cv2.ROTATE_180
#   None  (pas de rotation)
CAM_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

# ── Résolution POST-rotation (calculée automatiquement) ───────────────────────
# Utilisée par VideoWriter, YOLO, overlays, et ligne de comptage.
# Ne pas modifier manuellement — modifier CAM_NATIVE_WIDTH/HEIGHT et CAM_ROTATION.
if CAM_ROTATION in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
    CAM_WIDTH  = CAM_NATIVE_HEIGHT   # 720
    CAM_HEIGHT = CAM_NATIVE_WIDTH    # 1280
elif CAM_ROTATION == cv2.ROTATE_180:
    CAM_WIDTH  = CAM_NATIVE_WIDTH    # 1280
    CAM_HEIGHT = CAM_NATIVE_HEIGHT   # 720
else:
    CAM_WIDTH  = CAM_NATIVE_WIDTH    # 1280
    CAM_HEIGHT = CAM_NATIVE_HEIGHT   # 720

# ── Ligne de comptage ─────────────────────────────────────────────────────────
# Vue parallèle → ligne HORIZONTALE → on compare le centre Y du track.
# Calibrer avec check_camera.py, noter la valeur Y de la ligne verte,
# puis mettre à jour COUNT_LINE_Y ici.
COUNT_LINE_Y = 215      # ← à ajuster selon ton placement caméra
COUNT_LINE_MARGIN  = 15   # ← détecte le franchissement dans ±20px autour de la ligne

# ── YOLO ──────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.20

# ── SORT ──────────────────────────────────────────────────────────────────────
SORT_MAX_AGE  = 30
SORT_MIN_HITS = 2       # 1 = mode rapide pour véhicules rapides (>50km/h)
SORT_IOU      = 0.2     # permissif pour véhicules qui bougent vite entre frames

# ── Base de données ───────────────────────────────────────────────────────────
DB_DIR = "data"

# ── Watchdog caméra ───────────────────────────────────────────────────────────
CAM_WATCHDOG_TIMEOUT = 30   # secondes sans frame avant alerte

# ── Détection nuit/jour ───────────────────────────────────────────────────────
NIGHT_BRIGHTNESS_THRESHOLD = 40     # luminosité moyenne < seuil = nuit
NIGHT_CHECK_INTERVAL       = 30     # vérification toutes les N secondes
NIGHT_TRANSITION_FRAMES    = 10     # frames consécutives avant bascule

# ── MOG2 (mode nuit) ──────────────────────────────────────────────────────────
MOG2_HISTORY       = 200
MOG2_VAR_THRESHOLD = 25
MOG2_MIN_AREA      = 500

COUNTING_COOLDOWN_UP   = 0.5   # secondes
COUNTING_COOLDOWN_DOWN = 0.5

BBOX_SIMILARITY_RATIO = 0.5    #ratio min entre les deux areas pour etre un doublon
                               # 0.5 = la plus petite bbox doit faire au moins 50%
                               # de la plus grande
