# calibrate_line.py
import cv2
import argparse

def calibrate(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Impossible d'ouvrir : {video_path}")

    height, width = frame.shape[:2]
    line_y = height // 2  # position initiale au centre

    def on_trackbar(val):
        nonlocal line_y
        line_y = val

    cv2.namedWindow("Calibration ligne de comptage")
    cv2.createTrackbar("Ligne Y", "Calibration ligne de comptage", line_y, height - 1, on_trackbar)

    print("Ajuste la ligne avec le slider.")
    print("Vise la zone où les véhicules sont bien visibles (évite la zone obstruée par le toit).")
    print("Appuie sur 'q' pour valider et afficher la valeur Y.")

    while True:
        display = frame.copy()
        cv2.line(display, (0, line_y), (width, line_y), (0, 255, 0), 2)
        cv2.putText(display, f"Ligne Y = {line_y}  ({line_y/height*100:.0f}% du haut)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Calibration ligne de comptage", display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\n→ Utilise COUNT_LINE_Y = {line_y} dans ton script de comptage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Vidéo de référence")
    args = parser.parse_args()
    calibrate(args.video)