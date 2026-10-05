# check_camera.py — vérification placement caméra avec overlays
import time
import cv2
import argparse
from picamera2 import Picamera2
import config


def run(duration: int, output: str):
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={
            # Picamera2 reçoit toujours la résolution NATIVE (avant rotation)
            "size": (config.CAM_NATIVE_WIDTH, config.CAM_NATIVE_HEIGHT),
            "format": "RGB888"
        },
        controls={"FrameRate": config.CAM_FPS}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)

    # Dimensions post-rotation — utilisées pour VideoWriter et overlays
    w = config.CAM_WIDTH
    h = config.CAM_HEIGHT

    out = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.CAM_FPS,
        (w, h)
    )

    print(f"Enregistrement de {duration}s → {output}")
    print(f"Résolution native    : {config.CAM_NATIVE_WIDTH}x{config.CAM_NATIVE_HEIGHT}")
    print(f"Résolution effective : {w}x{h}  (post-rotation)")
    print(f"Ligne de comptage Y={config.COUNT_LINE_Y}  ({config.COUNT_LINE_Y/h*100:.0f}% du haut)")
    print(f"Rotation : {config.CAM_ROTATION}\n")

    start       = time.time()
    frame_count = 0

    while time.time() - start < duration:
        frame = cam.capture_array()

        # Rotation appliquée APRÈS capture
        if config.CAM_ROTATION is not None:
            frame = cv2.rotate(frame, config.CAM_ROTATION)

        # picamera2 retourne du RGB, OpenCV attend du BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_count += 1
        elapsed = time.time() - start

        # ── Grille de référence (tiers) ────────────────────────────────────────
        for x in [w // 3, 2 * w // 3]:
            cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
        for y in [h // 3, 2 * h // 3]:
            cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)

        # ── Ligne de comptage horizontale (COUNT_LINE_Y) ───────────────────────
        cv2.line(
            frame,
            (0, config.COUNT_LINE_Y),
            (w, config.COUNT_LINE_Y),
            (0, 255, 0), 2
        )
        cv2.putText(
            frame,
            f"Ligne Y={config.COUNT_LINE_Y}  ({config.COUNT_LINE_Y/h*100:.0f}% du haut)",
            (10, config.COUNT_LINE_Y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
        )

        # ── Réticule centre ────────────────────────────────────────────────────
        cx, cy = w // 2, h // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (100, 100, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (100, 100, 255), 1)

        # ── Overlays infos ─────────────────────────────────────────────────────
        cv2.putText(
            frame,
            f"{w}x{h}  @{config.CAM_FPS}fps",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1
        )
        cv2.putText(
            frame,
            f"{elapsed:.1f}s / {duration}s  (frame #{frame_count})",
            (10, h - 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1
        )

        out.write(frame)

        if frame_count % (config.CAM_FPS * 5) == 0:
            print(f"  {elapsed:.0f}s / {duration}s  —  {frame_count} frames")

    cam.stop()
    out.release()

    print(f"\nTerminé — {frame_count} frames enregistrées.")
    print(f"Récupère la vidéo avec :")
    print(f"  scp pi@traffic-pi.local:~/traffic-counter/{output} .")
    print(f"\nSi la ligne verte n'est pas au bon endroit :")
    print(f"  → Ajuste COUNT_LINE_Y dans config.py  (valeur actuelle : {config.COUNT_LINE_Y})")
    print(f"Si l'image est à l'envers :")
    print(f"  → Change CAM_ROTATION dans config.py  (valeur actuelle : {config.CAM_ROTATION})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vérification caméra — enregistrement vidéo avec overlays"
    )
    parser.add_argument(
        "--duration", type=int, default=30,
        help="Durée d'enregistrement en secondes (défaut: 30)"
    )
    parser.add_argument(
        "--output", default="camera_check.mp4",
        help="Nom du fichier de sortie (défaut: camera_check.mp4)"
    )
    args = parser.parse_args()
    run(args.duration, args.output)