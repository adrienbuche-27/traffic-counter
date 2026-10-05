import time
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n_ncnn_model", task="detect")

# Frame synthétique simulant ta résolution post-rotation
frame = np.random.randint(0, 255, (600, 400, 3), dtype=np.uint8)

# Warmup
model(frame, verbose=False)

# Benchmark 10 inférences
times = []
for i in range(10):
    t = time.time()
    model(frame, classes=[1,2,3,5,7], conf=0.4, verbose=False)
    times.append((time.time() - t) * 1000)

print(f"Latence moyenne : {sum(times)/len(times):.0f}ms")
print(f"Latence min     : {min(times):.0f}ms")
print(f"Latence max     : {max(times):.0f}ms")
print(f"FPS théoriques  : {1000/(sum(times)/len(times)):.1f}")