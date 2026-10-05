# live_counter.py — comptage temps réel, vue parallèle, threads découplés
import sys
import time
import signal
import logging
import sqlite3
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "sort"))
from sort import Sort
from ultralytics import YOLO
from picamera2 import Picamera2
import cv2

from night_detector import NightDetector
import config
from roi import ROI


# # ── Logging ────────────────────────────────────────────────────────────────────
# log_path = Path("data") / f"counter_{date.today().isoformat()}.log"

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s  %(levelname)-8s  %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler(log_path),
#     ]
# )


# log = logging.getLogger("traffic")

# ── Logging ────────────────────────────────────────────────────────────────────

# Handler console — tous les logs système (ALIVE, PERF, FPS, etc.)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

# Handler fichier — uniquement YOLO det, FRANCHISSEMENT, PASSAGE
# Rotation quotidienne automatique → counter_YYYY-MM-DD.log
class DailyDetectionHandler(logging.FileHandler):
    """
    FileHandler qui ne conserve que les lignes utiles au sanity_check :
    - YOLO det
    - FRANCHISSEMENT
    - PASSAGE ENREGISTRE
    - les lignes ===
    """
    ALLOWED = ("YOLO det", "FRANCHISSEMENT", "PASSAGE ENREGISTRE", "===")

    def emit(self, record):
        msg = record.getMessage()
        if any(tag in msg for tag in self.ALLOWED):
            super().emit(record)

def get_daily_log_path() -> str:
    return str(
        Path("data") /
        f"counter_{date.today().isoformat()}.log"
    )

Path(get_daily_log_path()).touch(exist_ok=True)

daily_handler = DailyDetectionHandler(get_daily_log_path())
daily_handler.setLevel(logging.INFO)
daily_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler, daily_handler]
)
log = logging.getLogger("traffic")

TARGET_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── Base de données ────────────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    db_path = Path(config.DB_DIR) / f"traffic_{date.today().isoformat()}.db"
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS passages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            class     TEXT    NOT NULL,
            direction TEXT    NOT NULL,
            track_id  INTEGER NOT NULL,
            bbox_w    INTEGER DEFAULT 0,
            bbox_h    INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    log.info(f"DB ouverte : {db_path}")
    return conn

def log_passage(conn, db_lock, cls, direction, track_id,
                bbox_w: int = 0, bbox_h: int = 0):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    with db_lock:
        conn.execute(
            "INSERT INTO passages "
            "(timestamp, class, direction, track_id, bbox_w, bbox_h) "
            "VALUES (?,?,?,?,?,?)",
            (ts, cls, direction, track_id, bbox_w, bbox_h)
        )
        conn.commit()
    log.info("=" * 50)
    log.info(
        f"PASSAGE ENREGISTRE  {cls:<12}  {direction}  "
        f"track#{track_id}  bbox={bbox_w}x{bbox_h}  @ {ts}"
    )
    log.info("=" * 50)

# ── Watchdog caméra ────────────────────────────────────────────────────────────
class CameraWatchdog:
    def __init__(self, timeout: int = config.CAM_WATCHDOG_TIMEOUT):
        self.timeout    = timeout
        self.last_frame = time.time()
        self._alerted   = False

    def heartbeat(self):
        self.last_frame = time.time()
        self._alerted   = False

    def check(self):
        elapsed = time.time() - self.last_frame
        if elapsed > self.timeout and not self._alerted:
            log.error(
                f"WATCHDOG — aucune frame depuis {elapsed:.0f}s. "
                f"Caméra décrochée ?"
            )
            self._alerted = True

# ── Initialisation caméra ──────────────────────────────────────────────────────
def init_camera() -> Picamera2:
    log.info("Initialisation caméra CSI...")
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={
            "size": (config.CAM_NATIVE_WIDTH, config.CAM_NATIVE_HEIGHT),
            "format": "RGB888"
        },
        controls={"FrameRate": config.CAM_FPS}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)
    w, h = config.CAM_WIDTH, config.CAM_HEIGHT
    log.info(
        f"Caméra démarrée  native={config.CAM_NATIVE_WIDTH}x{config.CAM_NATIVE_HEIGHT}  "
        f"post-rotation={w}x{h}  @{config.CAM_FPS}fps"
    )
    return cam

# ── Thread de capture ──────────────────────────────────────────────────────────
def capture_thread(cam, frame_queue, watchdog, stop_event):
    """
    Capture les frames en continu, applique la rotation,
    et pousse dans la queue. Si la queue est pleine, droppe
    la frame la plus ancienne pour ne jamais accumuler de retard.
    """
    fps_counter = 0
    fps_timer   = time.time()

    while not stop_event.is_set():
        try:
            frame        = cam.capture_array()
            capture_time = time.time()
            watchdog.heartbeat()
        except Exception as e:
            log.error(f"[CAPTURE] Erreur : {e}")
            watchdog.check()
            time.sleep(0.1)
            continue

        # Rotation appliquée dès la capture
        if config.CAM_ROTATION is not None:
            frame = cv2.rotate(frame, config.CAM_ROTATION)

        # Drop la frame la plus ancienne si queue pleine
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
                # log.info("[CAPTURE] Queue pleine — frame ancienne droppée")
            except queue.Empty:
                pass

        frame_queue.put((frame, capture_time))

        fps_counter += 1
        if fps_counter >= 30:
            elapsed     = time.time() - fps_timer
            # log.info(f"[CAPTURE] {fps_counter / elapsed:.1f} FPS capturés")
            fps_counter = 0
            fps_timer   = time.time()

    log.info("[CAPTURE] Thread arrêté")

# ── Thread de traitement ───────────────────────────────────────────────────────
def processing_thread(frame_queue, conn, db_lock, stop_event):
    """
    Consomme les frames de la queue, fait tourner YOLO + SORT,
    enregistre les passages. Tourne aussi vite que le CPU le permet.
    """
    model = YOLO("yolov8n_ncnn_model", task="detect")
    log.info("[PROCESSING] Modèle YOLOv8n NCNN chargé")

    roi = ROI()
    if roi.enabled:
        log.info(f"[ROI] Masque polygonal activé — bbox={roi.bbox}")
    else:
        log.info("[ROI] Aucune ROI définie — image complète utilisée")

    tracker = Sort(
        max_age=config.SORT_MAX_AGE,
        min_hits=config.SORT_MIN_HITS,
        iou_threshold=config.SORT_IOU
    )
    log.info(
        f"[PROCESSING] Tracker SORT  max_age={config.SORT_MAX_AGE}  "
        f"min_hits={config.SORT_MIN_HITS}  iou={config.SORT_IOU}"
    )

    night_detector           = NightDetector()
    is_night                 = False
    night_check_timer        = time.time()
    night_transition_counter = 0

    track_history   = defaultdict(list)
    counted_ids     = set()
    track_class_map = {}
    current_day     = date.today()

    frame_count     = 0
    fps_counter     = 0
    fps_timer       = time.time()
    last_stats_log  = time.time()
    latency_history = []

    # Résolution post-rotation — une seule fois depuis config
    frame_w, frame_h = config.CAM_WIDTH, config.CAM_HEIGHT
    last_counted_time = {"↑": 0.0, "↓": 0.0}

    while not stop_event.is_set():

        try:
            frame, capture_time = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        fps_counter += 1

        # ── Rotation DB quotidienne ────────────────────────────────────────────
        today = date.today()
        if today != current_day:
            current_day = today
            counted_ids.clear()
            track_history.clear()
            track_class_map.clear()
            latency_history.clear()
            log.info(f"Rotation DB — nouveau fichier : traffic_{today}.db")

        now = time.time()

        # ── FPS toutes les 30 frames ───────────────────────────────────────────
        if fps_counter >= 30:
            elapsed  = time.time() - fps_timer
            proc_fps = fps_counter / elapsed
            # log.info(
            #     f"[PROCESSING] {proc_fps:.1f} FPS traités  "
            #     f"queue={frame_queue.qsize()}/{frame_queue.maxsize}"
            # )
            fps_counter = 0
            fps_timer   = time.time()

        # ── Signe de vie + métriques toutes les 30s ───────────────────────────
        if now - last_stats_log >= 30:
            mode = "NUIT" if is_night else "JOUR"
            lag  = frame_queue.qsize()
            if latency_history:
                avg_lat = sum(latency_history) / len(latency_history)
                max_lat = max(latency_history)
                eff_fps = 1000 / avg_lat if avg_lat > 0 else 0
                status  = (
                    "excellent"  if avg_lat < 200  else
                    "acceptable" if avg_lat < 500  else
                    "lent"       if avg_lat < 1000 else
                    "critique"
                )
                # log.info(
                #     f"[ALIVE] mode={mode}  frames={frame_count}  "
                #     f"passages={len(counted_ids)}  queue={lag}"
                # )
                # log.info(
                #     f"[PERF]  latence moy={avg_lat:.0f}ms  "
                #     f"max={max_lat:.0f}ms  "
                #     f"FPS effectifs={eff_fps:.1f}  "
                #     f"statut={status}"
                # )
            else:
                log.info(
                    f"[ALIVE] mode={mode}  frames={frame_count}  "
                    f"passages={len(counted_ids)}  queue={lag}"
                )
            last_stats_log = now

        # ── Détection automatique jour/nuit ───────────────────────────────────
        if now - night_check_timer >= config.NIGHT_CHECK_INTERVAL:
            night_check_timer = now
            gray_frame  = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            brightness  = float(np.mean(gray_frame))
            was_night   = is_night

            if brightness < config.NIGHT_BRIGHTNESS_THRESHOLD:
                night_transition_counter = min(
                    night_transition_counter + 1,
                    config.NIGHT_TRANSITION_FRAMES
                )
            else:
                night_transition_counter = max(
                    night_transition_counter - 1, 0
                )

            is_night = (
                night_transition_counter >= config.NIGHT_TRANSITION_FRAMES // 2
            )

            log.debug(
                f"Luminosité={brightness:.1f}  "
                f"counter={night_transition_counter}  "
                f"mode={'NUIT' if is_night else 'JOUR'}"
            )

            if is_night != was_night:
                mode_str = "NUIT (MOG2)" if is_night else "JOUR (YOLO)"
                log.info(f"Bascule mode → {mode_str}  (brightness={brightness:.1f})")
                if not is_night:
                    night_detector.reset()
                    track_history.clear()
                    track_class_map.clear()
                    counted_ids.clear()
                    latency_history.clear()

        # ══════════════════════════════════════════════════════════════════════
        # MODE NUIT — MOG2
        # ══════════════════════════════════════════════════════════════════════
        if is_night:
            passages = night_detector.process(frame)
            for p in passages:
                log_passage(conn, db_lock, p["class"], p["direction"], -1)
            continue

        # ══════════════════════════════════════════════════════════════════════
        # MODE JOUR — YOLO + SORT
        # ══════════════════════════════════════════════════════════════════════
        yolo_start = time.time()

        frame_roi, offset = roi.apply(frame)

        results = model(
            frame_roi,
            classes=list(TARGET_CLASSES.keys()),
            conf=config.CONF_THRESHOLD,
            # imgsz=320,
            verbose=False
        )[0]

        inference_time = time.time()
        latency_ms     = (inference_time - capture_time) * 1000
        yolo_ms        = (inference_time - yolo_start)   * 1000

        latency_history.append(latency_ms)
        if len(latency_history) > 30:
            latency_history.pop(0)

        log.debug(
            f"[LATENCY] frame#{frame_count}  "
            f"capture→inference={latency_ms:.0f}ms  "
            f"yolo={yolo_ms:.0f}ms  "
            f"queue={frame_queue.qsize()}"
        )

        dets       = []
        dets_class = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1, y1, x2, y2 = roi.restore_coords(x1, y1, x2, y2, offset)
            score   = float(box.conf)
            cls_id  = int(box.cls)
            label   = TARGET_CLASSES[cls_id]
            dets.append([x1, y1, x2, y2, score])
            dets_class.append(label)
            log.info(
                f"  YOLO det  {label:<12}  conf={score:.2f}  "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
            )

        if len(results.boxes) > 0:
            log.debug(
                f"Frame #{frame_count} — {len(results.boxes)} détection(s) YOLO"
            )

        # ── Tracking SORT ──────────────────────────────────────────────────────
        tracked = tracker.update(
            np.array(dets) if dets else np.empty((0, 5))
        )

        for t in tracked:
            tx1, ty1, tx2, ty2, tid = map(int, t)
            tid = int(tid)
            cx  = (tx1 + tx2) // 2
            cy  = (ty1 + ty2) // 2

            if dets:
                best_idx = min(
                    range(len(dets)),
                    key=lambda i: abs(cx - (dets[i][0] + dets[i][2]) / 2)
                                + abs(cy - (dets[i][1] + dets[i][3]) / 2)
                )
                track_class_map[tid] = dets_class[best_idx]

            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)

            best_cls    = track_class_map.get(tid)
            positions_y = [p[1] for p in track_history[tid]]
            hits        = len(positions_y)

            log.debug(
                f"  Track #{tid:<4} {str(best_cls):<12}  "
                f"cy={cy}  hits={hits}  "
                f"counted={'oui' if tid in counted_ids else 'non'}"
            )

            # ── Franchissement ligne horizontale ──────────────────────────────
            if tid not in counted_ids and hits >= 2 and best_cls:
                prev_y = positions_y[-2]
                curr_y = positions_y[-1]
                line_top    = config.COUNT_LINE_Y - config.COUNT_LINE_MARGIN
                line_bottom = config.COUNT_LINE_Y + config.COUNT_LINE_MARGIN

                crossed = (
                    (prev_y < line_bottom and curr_y >= line_top) or
                    (prev_y > line_top    and curr_y <= line_bottom)
                )

                if crossed:
                    direction = "↓" if curr_y > prev_y else "↑"
                    cooldown  = config.COUNTING_COOLDOWN_DOWN if direction == "↓" else config.COUNTING_COOLDOWN_UP

                    # Calcule la taille de la bbox au moment du franchissement
                    bbox_w = int(tx2 - tx1)
                    bbox_h = int(ty2 - ty1)

                    if now - last_counted_time[direction] >= cooldown:
                        log.info(
                            f"FRANCHISSEMENT  track#{tid}  {best_cls}  "
                            f"prev_y={prev_y}  curr_y={curr_y}  "
                            f"ligne={config.COUNT_LINE_Y}  {direction}"
                        )
                        log_passage(conn, db_lock, best_cls, direction, tid,
                                    bbox_w=bbox_w, bbox_h=bbox_h)
                        counted_ids.add(tid)
                        last_counted_time[direction] = now
                    else:
                        log.info(f"  COOLDOWN {direction} actif — passage ignoré")
                else:
                    if abs(cy - config.COUNT_LINE_Y) < 100:
                        log.debug(
                            f"  Track #{tid} proche ligne  "
                            f"prev_y={prev_y}  curr_y={curr_y}  "
                            f"ligne={config.COUNT_LINE_Y}"
                        )

        # Nettoyage tracks disparus
        active_ids = {int(t[4]) for t in tracked}
        for tid in list(track_history.keys()):
            if tid not in active_ids:
                log.debug(f"Track #{tid} disparu")
                del track_history[tid]
                track_class_map.pop(tid, None)

    # ── Résumé final ───────────────────────────────────────────────────────────
    if latency_history:
        avg_lat = sum(latency_history) / len(latency_history)
        log.info("=" * 50)
        log.info("Arrêt processing thread")
        log.info(f"  Frames traitées  : {frame_count}")
        log.info(f"  Passages comptés : {len(counted_ids)}")
        log.info(f"  Latence moyenne  : {avg_lat:.0f}ms")
        log.info(f"  FPS effectifs    : {1000/avg_lat:.1f}")
        log.info("=" * 50)
    else:
        log.info(
            f"Arrêt processing thread — {frame_count} frames  "
            f"{len(counted_ids)} passages"
        )
    log.info("[PROCESSING] Thread arrêté")

# ── Point d'entrée ─────────────────────────────────────────────────────────────
def run():
    # w, h = config.rotated_size()
    log.info("=" * 50)
    log.info("Démarrage traffic-counter (threadé, vue parallèle)")
    log.info(f"  Résolution native  : {config.CAM_NATIVE_WIDTH}x{config.CAM_NATIVE_HEIGHT}")
    log.info(f"  Rotation           : {config.CAM_ROTATION}")
    log.info(f"  Résolution sortie  : {config.CAM_WIDTH}x{config.CAM_HEIGHT}")
    log.info(f"  Ligne Y            : {config.COUNT_LINE_Y}")
    log.info(f"  Conf YOLO          : {config.CONF_THRESHOLD}")
    log.info(f"  SORT min_hits      : {config.SORT_MIN_HITS}")
    log.info(f"  SORT max_age       : {config.SORT_MAX_AGE}")
    log.info("=" * 50)

    cam      = init_camera()
    watchdog = CameraWatchdog()
    conn     = get_db()
    db_lock  = threading.Lock()

    frame_queue = queue.Queue(maxsize=3)
    stop_event  = threading.Event()

    t_capture = threading.Thread(
        target=capture_thread,
        args=(cam, frame_queue, watchdog, stop_event),
        daemon=True,
        name="capture"
    )
    t_processing = threading.Thread(
        target=processing_thread,
        args=(frame_queue, conn, db_lock, stop_event),
        daemon=True,
        name="processing"
    )

    t_capture.start()
    t_processing.start()
    log.info("Threads capture et processing démarrés")

    def on_stop(sig, frame):
        log.info("Signal arrêt reçu — fermeture propre...")
        stop_event.set()

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT,  on_stop)

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        stop_event.set()
        t_capture.join(timeout=3)
        t_processing.join(timeout=5)
        cam.stop()
        conn.close()
        log.info("Arrêt propre terminé")

if __name__ == "__main__":
    run()