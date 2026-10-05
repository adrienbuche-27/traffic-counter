# count_phase2.py — comptage sur vidéo fichier, vue parallèle, ligne Y
# Note : ce script attend une vidéo déjà orientée correctement.
# Si ta vidéo vient de check_camera.py, la rotation est déjà appliquée.
# Si ta vidéo vient directement de la caméra (non tournée), passe --rotate.
import cv2
import sqlite3
import argparse
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import config

sys.path.insert(0, str(Path(__file__).parent / "sort"))
from sort import Sort
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
TARGET_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

COLORS = {
    "bicycle":    (0, 200, 100),
    "car":        (50, 150, 255),
    "motorcycle": (0, 165, 255),
    "bus":        (180, 60, 255),
    "truck":      (60, 60, 200),
}

DB_PATH = Path(__file__).parent / "traffic.db"


# ── Base de données ────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS passages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            class     TEXT    NOT NULL,
            direction TEXT    NOT NULL,
            track_id  INTEGER NOT NULL
        )
    """)
    conn.commit()
    return conn

def log_passage(conn, cls, direction, track_id):
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn.execute(
        "INSERT INTO passages (timestamp, class, direction, track_id) VALUES (?,?,?,?)",
        (ts, cls, direction, track_id)
    )
    conn.commit()
    print(f"  ✓ Passage — {ts}  {cls:<12} {direction}  (track #{track_id})")


# ── Traitement vidéo ───────────────────────────────────────────────────────────
def process(input_path: str, output_path: str, count_line_y: int,
            conf: float, apply_rotation: bool):

    model   = YOLO("yolov8n_ncnn_model", task="detect")
    tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.2)
    conn    = init_db()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir : {input_path}")

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Dimensions de sortie selon rotation demandée
    if apply_rotation:
        w, h = raw_h, raw_w
        print(f"Rotation activée : {raw_w}x{raw_h} → {w}x{h}")
    else:
        w, h = raw_w, raw_h
        print(f"Pas de rotation — vidéo déjà orientée : {w}x{h}")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    track_history   = defaultdict(list)
    counted_ids     = set()
    track_class_map = {}
    frame_count     = 0
    total_counts    = defaultdict(int)

    print(f"Ligne de comptage Y={count_line_y}  ({count_line_y/h*100:.0f}% du haut)")
    print(f"Traitement de {input_path} @ {fps:.1f}fps...\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Rotation optionnelle — seulement si la vidéo n'est pas déjà tournée
        if apply_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_count += 1

        # ── Détection YOLO ─────────────────────────────────────────────────────
        results = model(
            frame,
            classes=list(TARGET_CLASSES.keys()),
            conf=conf,
            verbose=False
        )[0]

        dets       = []
        dets_class = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            score   = float(box.conf)
            cls_id  = int(box.cls)
            dets.append([x1, y1, x2, y2, score])
            dets_class.append(TARGET_CLASSES[cls_id])

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

            best_cls = track_class_map.get(tid)

            track_history[tid].append((cx, cy))
            if len(track_history[tid]) > 30:
                track_history[tid].pop(0)

            positions_y = [p[1] for p in track_history[tid]]

            # ── Franchissement ligne horizontale Y ─────────────────────────────
            if tid not in counted_ids and len(positions_y) >= 2 and best_cls:
                prev_y = positions_y[-2]
                curr_y = positions_y[-1]
                crossed = (
                    (prev_y < count_line_y <= curr_y) or
                    (prev_y > count_line_y >= curr_y)
                )
                if crossed:
                    direction = "↓" if curr_y > prev_y else "↑"
                    log_passage(conn, best_cls, direction, tid)
                    counted_ids.add(tid)
                    total_counts[best_cls] += 1
                else:
                    if abs(cy - count_line_y) < 100:
                        print(
                            f"  Track #{tid} proche ligne  "
                            f"prev_y={prev_y}  curr_y={curr_y}  "
                            f"ligne={count_line_y}  (pas encore franchi)"
                        )

            # ── Dessin ─────────────────────────────────────────────────────────
            color = COLORS.get(track_class_map.get(tid, ""), (200, 200, 200))
            label = f"#{tid} {track_class_map.get(tid, '?')}"
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, 2)
            cv2.putText(frame, label, (tx1, ty1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.circle(frame, (cx, cy), 4, color, -1)

        # ── Ligne de comptage horizontale ──────────────────────────────────────
        cv2.line(frame, (0, count_line_y), (w, count_line_y), (0, 255, 0), 2)
        cv2.putText(
            frame, "Ligne de comptage",
            (10, count_line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # ── Compteurs overlay ──────────────────────────────────────────────────
        y_offset = 30
        for cls, cnt in sorted(total_counts.items()):
            cv2.putText(
                frame, f"{cls}: {cnt}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                COLORS.get(cls, (255, 255, 255)), 2
            )
            y_offset += 25

        out.write(frame)

    cap.release()
    out.release()
    conn.close()

    print(f"\n── Résumé ({frame_count} frames) ──")
    for cls, cnt in sorted(total_counts.items()):
        print(f"  {cls:<12}: {cnt} passage(s)")
    print(f"\nVidéo annotée  → {output_path}")
    print(f"Base de données → {DB_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 — Comptage trafic sur fichier vidéo (vue parallèle, ligne Y)"
    )
    parser.add_argument("input",       help="Vidéo source")
    parser.add_argument("--output",    default="output_counted.mp4")
    parser.add_argument("--line-y",    type=int, required=False,
                        help="Position Y de la ligne de comptage", default = config.COUNT_LINE_Y)
    parser.add_argument("--conf",      type=float, default=0.4)
    parser.add_argument("--rotate",    action="store_true",
                        help="Appliquer rotation 90° (si vidéo non encore tournée)")
    args = parser.parse_args()

    process(args.input, args.output, args.line_y, args.conf, args.rotate)