# Traffic Counter

Real-time road traffic counter for a **Raspberry Pi with a CSI camera**.

The app watches a road, detects vehicles with **YOLOv8n (NCNN export)**, tracks them
with **SORT**, and logs a passage each time a vehicle crosses a horizontal
**counting line**. Passages are stored in daily SQLite databases and shown on a
**Flask web dashboard**. A **sanity check** tool (optionally helped by the Claude API)
reads the detection logs back to find missed passages and duplicates.

```
 Pi Camera (Picamera2) ──► capture thread ──► queue ──► processing thread
                                                          │
                          day:   ROI mask ─► YOLOv8n NCNN ─► SORT ─► line crossing
                          night: MOG2 headlight detector (see Known issues)
                                                          │
                                         data/traffic_YYYY-MM-DD.db   (passages)
                                         data/counter_YYYY-MM-DD.log  (detections)
                                                          │
                              dashboard/app.py (Flask :8080) + sanity_check.py
```

---

## Table of contents

1. [Hardware](#1-hardware)
2. [Repository layout](#2-repository-layout)
3. [Installation on the Raspberry Pi](#3-installation-on-the-raspberry-pi)
4. [Camera placement and calibration](#4-camera-placement-and-calibration)
5. [Configuration reference (`config.py`)](#5-configuration-reference-configpy)
6. [Running the counter](#6-running-the-counter)
7. [Dashboard](#7-dashboard)
8. [Sanity check](#8-sanity-check)
9. [Data and database schema](#9-data-and-database-schema)
10. [Other tools](#10-other-tools)
11. [Known issues and caveats](#11-known-issues-and-caveats)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Hardware

- A Raspberry Pi 4 or 5 (64-bit) is recommended, because YOLO inference runs on the CPU
- A Raspberry Pi **CSI camera module** that works with `libcamera` / `Picamera2`
- A microSD card (32 GB or more) running **Raspberry Pi OS Bookworm 64-bit**
- A good power supply and some cooling. `run_counter.sh` adds a pause between
  runs so the Pi can cool down.
- Network access (Wi-Fi or Ethernet) for the dashboard and SSH

The camera should look **along the road** (parallel view), so vehicles move
**up and down** in the image. The counting line is horizontal.

---

## 2. Repository layout

| File | Purpose |
|---|---|
| `config.py` | **Single source of truth** for all settings (camera, line, YOLO, SORT, night mode…) |
| `live_counter.py` | Main app: live capture, detection, tracking, counting, DB writes |
| `run_counter.sh` | Wrapper that restarts `live_counter.py` in cycles (run N minutes, pause, restart) |
| `traffic-counter.service` | systemd unit that runs `run_counter.sh` at boot |
| `roi.py` | Optional polygon Region Of Interest (mask + crop before YOLO) |
| `night_detector.py` | Night mode: headlight detection with MOG2 background subtraction |
| `check_camera.py` | Records a short video with the counting line and grid drawn on it, to check camera placement |
| `calibrate_line.py` | Interactive slider (needs a display) to pick `COUNT_LINE_Y` on a video frame |
| `calibrate_brightness.py` | Prints the mean image brightness every 5 s, to tune the day/night threshold |
| `count_phase2.py` | Offline counting on a video file, writes an annotated video and `traffic.db` |
| `benchmark.py` | Measures YOLO NCNN inference speed on the Pi |
| `sanity_check.py` | Compares detection logs to the DB: finds missed passages and duplicates |
| `migrate_add_bbox.py` / `migrate_add_confidence.py` | One-shot DB schema migrations |
| `dashboard/app.py` | Flask dashboard and REST API (port 8080) |
| `dashboard/templates/` | `index.html` (today + sanity check UI), `history.html` (multi-day history) |
| `videos/` | Sample videos for offline tests |

Some things are **not in git** (see `.gitignore`) and must be created on the Pi:
`sort/` (the SORT tracker), `yolov8n_ncnn_model/` (the exported model), `data/`
contents (`*.db`, `*.log`, `*.json`, `roi.json`) and the Python virtual env.

---

## 3. Installation on the Raspberry Pi

The steps below use the paths from the current scripts:
**`/home/abuche/Documents/traffic-counter`** for the project and a virtualenv
named **`traffic`** inside it. If you use another user or path, see
[3.8 Adapt the hard-coded paths](#38-adapt-the-hard-coded-paths).

### 3.1 Flash and prepare the OS

1. Flash **Raspberry Pi OS (64-bit) Bookworm** with Raspberry Pi Imager.
   In the imager settings, set the hostname (for example `traffic-pi`), the user,
   Wi-Fi, and turn on SSH.
2. Boot, connect over SSH, and update:

   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo reboot
   ```

### 3.2 Enable and test the camera

On Bookworm the CSI camera is detected automatically through `libcamera`.

```bash
sudo apt install -y python3-picamera2 libcamera-apps
rpicam-hello --list-cameras        # older images: libcamera-hello --list-cameras
rpicam-still -o test.jpg           # take a test picture
```

If no camera is listed, check the ribbon cable and `/boot/firmware/config.txt`
(`camera_auto_detect=1`, or a `dtoverlay=` line for your sensor).

### 3.3 System packages

```bash
sudo apt install -y git python3-venv python3-pip python3-opencv \
                    libatlas-base-dev sqlite3
```

### 3.4 Clone the project

```bash
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/adrienbuche-27/traffic-counter.git
cd traffic-counter
mkdir -p data          # required: live_counter.py writes its log and DBs here
```

### 3.5 Python virtual environment

`picamera2` comes from apt and can't be installed with pip easily, so the venv
**must see the system packages**:

```bash
python3 -m venv --system-site-packages traffic
source traffic/bin/activate
pip install --upgrade pip
pip install ultralytics ncnn flask anthropic filterpy scikit-image lap
```

| Package | Used by |
|---|---|
| `ultralytics`, `ncnn` | YOLOv8 detection with the NCNN backend |
| `opencv` (apt `python3-opencv`, or pulled by ultralytics) | Image processing, MOG2, video I/O |
| `picamera2` (apt) | Camera capture |
| `filterpy`, `scikit-image`, `lap` | Dependencies of the SORT tracker |
| `flask` | Dashboard |
| `anthropic` | Claude API in `sanity_check.py` (imported at the top, so it's needed even with `--no-claude`) |

Check that the venv can see the camera:

```bash
python -c "from picamera2 import Picamera2; print('picamera2 OK')"
```

### 3.6 Install the SORT tracker

The code imports `Sort` from a local `sort/` folder (`sys.path.insert(..., "sort")`):

```bash
cd ~/Documents/traffic-counter
git clone https://github.com/abewley/sort.git sort
```

> `sort/sort.py` calls `matplotlib.use('TkAgg')` when it is imported. On a headless Pi
> this can fail. If it does, comment out that line, or replace it with
> `matplotlib.use('Agg')`.

### 3.7 Export the YOLOv8n model to NCNN

The scripts load the model from the folder `yolov8n_ncnn_model/` in the project root:

```bash
source traffic/bin/activate
yolo export model=yolov8n.pt format=ncnn      # downloads yolov8n.pt and creates yolov8n_ncnn_model/
python benchmark.py                            # optional: check the inference speed
```

### 3.8 Adapt the hard-coded paths

Some paths and the user name are hard-coded. Make them match your setup:

| File | What to check |
|---|---|
| `run_counter.sh` | `LOG`, `VENV`, `SCRIPT`, `WORKDIR` (now `/home/abuche/Documents/traffic-counter`) |
| `traffic-counter.service` | `User=` / `Group=` (now `pi`, which **doesn't match** `/home/abuche/...`), `WorkingDirectory=`, `ExecStart=` |
| `dashboard/app.py` | `sys.path.insert(0, "/home/abuche/traffic-counter")`, which points to a folder **without** `Documents/`. It still works because the project root is also added to the path, but it's safer to fix it. |
| `check_camera.py` | Only the `scp pi@traffic-pi.local:...` hint it prints at the end |

Make the runner executable:

```bash
chmod +x run_counter.sh
```

---

## 4. Camera placement and calibration

All coordinates are in the **image after rotation** (`CAM_WIDTH x CAM_HEIGHT`).
With the default settings this is 600 x 400 px: the native 400 x 600 image
rotated 90° counter-clockwise.

### 4.1 Orientation: `CAM_ROTATION`

1. Mount the camera so that it looks along the road.
2. Record a check video:

   ```bash
   source traffic/bin/activate
   python check_camera.py --duration 30 --output camera_check.mp4
   ```

3. Copy it to your computer and watch it:

   ```bash
   scp <user>@traffic-pi.local:~/Documents/traffic-counter/camera_check.mp4 .
   ```

4. If the image is rotated or upside down, change `CAM_ROTATION` in `config.py`
   (`cv2.ROTATE_90_CLOCKWISE`, `cv2.ROTATE_90_COUNTERCLOCKWISE`, `cv2.ROTATE_180`
   or `None`) and record again.

### 4.2 Counting line: `COUNT_LINE_Y`

The green line in `camera_check.mp4` is the current `COUNT_LINE_Y`. Put it
across the road, where vehicles are clearly visible and not hidden by anything.

To pick the value with a slider (needs a screen, a VNC session, or a desktop computer):

```bash
python calibrate_line.py camera_check.mp4
# move the slider, press 'q', then copy the printed value
```

Then set `COUNT_LINE_Y` (and if needed `COUNT_LINE_MARGIN`) in `config.py`.

Direction convention:
- **`↑`**: Y goes down in the image (vehicle comes **towards** the camera)
- **`↓`**: Y goes up in the image (vehicle goes **away** from the camera)

### 4.3 Region of interest (optional): `data/roi.json`

If `data/roi.json` exists, YOLO only sees the inside of a polygon: the rest is
masked and the image is cropped to the polygon's bounding box. Inference is faster
and you get fewer false positives (parked cars, sidewalk…). There is no tool to
make this file, so write it by hand, with coordinates in the rotated image:

```json
{
  "points": [[40, 120], [560, 120], [600, 330], [0, 330]],
  "img_width": 600,
  "img_height": 400
}
```

`img_width` / `img_height` must be the post-rotation size (`CAM_WIDTH`, `CAM_HEIGHT`).
Delete the file to go back to using the full image. At startup the log shows
`[ROI] Masque polygonal activé` or `[ROI] Aucune ROI définie`.

### 4.4 Day/night threshold: `NIGHT_BRIGHTNESS_THRESHOLD`

Run this during the day **and** at night and note the values:

```bash
python calibrate_brightness.py      # Ctrl+C to stop
```

Set `NIGHT_BRIGHTNESS_THRESHOLD` between the darkest "day" value and the brightest
"night" value. Read [Known issues](#11-known-issues-and-caveats) about night mode first.

---

## 5. Configuration reference (`config.py`)

`config.py` at the project root is used by **every** script, including the
dashboard. (`dashboard/config.py` is an older copy and is not used.)

### Camera

| Setting | Default | Description |
|---|---|---|
| `CAM_NATIVE_WIDTH` | `400` | Sensor capture width sent to Picamera2 (before rotation) |
| `CAM_NATIVE_HEIGHT` | `600` | Sensor capture height (before rotation) |
| `CAM_FPS` | `10` | Requested frame rate |
| `CAM_ROTATION` | `cv2.ROTATE_90_COUNTERCLOCKWISE` | Rotation applied to every frame right after capture |
| `CAM_WIDTH` / `CAM_HEIGHT` | *computed* | Size after rotation. **Don't edit these:** change the native size or the rotation instead. |
| `CAM_WATCHDOG_TIMEOUT` | `30` | Seconds without a frame before a `WATCHDOG` error is logged |

### Counting

| Setting | Default | Description |
|---|---|---|
| `COUNT_LINE_Y` | `215` | Y position of the horizontal counting line (post-rotation pixels) |
| `COUNT_LINE_MARGIN` | `15` | A crossing is detected inside a ±margin band around the line. Make it larger for fast vehicles at low FPS. |
| `COUNTING_COOLDOWN_UP` | `0.5` | Minimum seconds between two `↑` passages (against double counts) |
| `COUNTING_COOLDOWN_DOWN` | `0.5` | Same for `↓` |
| `BBOX_SIMILARITY_RATIO` | `0.5` | Minimum area ratio between two bboxes to call them duplicates (sanity check) |

### Detection (YOLO)

| Setting | Default | Description |
|---|---|---|
| `CONF_THRESHOLD` | `0.20` | Minimum YOLO confidence. Lower it to catch more vehicles, at the cost of more false positives. |
| `YOLO_IMGSZ` | `320` | Not used right now (the `imgsz` argument is commented out in `live_counter.py`) |

Counted classes (COCO): `bicycle`, `car`, `motorcycle`, `bus`, `truck`.

### Tracking (SORT)

| Setting | Default | Description |
|---|---|---|
| `SORT_MAX_AGE` | `30` | Frames a track stays alive without a matching detection |
| `SORT_MIN_HITS` | `2` | Detections needed before a track is confirmed (`1` = fast mode for vehicles over 50 km/h) |
| `SORT_IOU` | `0.2` | IoU threshold to match detections to tracks (kept low for fast-moving vehicles) |

### Day/night and night mode (MOG2)

| Setting | Default | Description |
|---|---|---|
| `NIGHT_BRIGHTNESS_THRESHOLD` | `40` | Mean gray level below this counts as night |
| `NIGHT_CHECK_INTERVAL` | `30` | Seconds between brightness checks |
| `NIGHT_TRANSITION_FRAMES` | `10` | Hysteresis counter: the mode switches to night when it reaches half of this value |
| `MOG2_HISTORY` | `200` | MOG2 background history |
| `MOG2_VAR_THRESHOLD` | `25` | MOG2 variance threshold |
| `MOG2_MIN_AREA` | `500` | Minimum blob area (px²) for a headlight |

### Storage

| Setting | Default | Description |
|---|---|---|
| `DB_DIR` | `"data"` | Folder for the DBs, **relative to the current directory**. Always run the scripts from the project root. |

### Cycle settings (`run_counter.sh`)

| Variable | Default | Description |
|---|---|---|
| `RUN_MINUTES` | `10` | How long each `live_counter.py` run lasts before `timeout` stops it with SIGTERM |
| `COOLDOWN_SECONDS` | `10` | Pause between two runs |

---

## 6. Running the counter

### 6.1 Manual run (for testing)

```bash
cd ~/Documents/traffic-counter
source traffic/bin/activate
python live_counter.py           # Ctrl+C stops it cleanly
```

At startup it prints the active configuration. Then each counted vehicle shows up as:

```
FRANCHISSEMENT  track#12  car  prev_y=205  curr_y=221  ligne=215  ↓
PASSAGE ENREGISTRE  car           ↓  track#12  bbox=84x61  @ 2026-10-05 14:02:11
```

### 6.2 Run as a service (start at boot)

`run_counter.sh` runs `live_counter.py` for `RUN_MINUTES`, stops it cleanly,
waits `COOLDOWN_SECONDS`, and starts it again, forever. The systemd unit starts
this loop at boot and restarts it if it crashes.

```bash
# After fixing User/Group/paths in the file (see 3.8):
sudo cp traffic-counter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now traffic-counter

# Monitor
sudo systemctl status traffic-counter
journalctl -u traffic-counter -f                   # live console output
tail -f data/runner.log                            # cycle start/stop log
```

Stop or restart:

```bash
sudo systemctl stop traffic-counter
sudo systemctl restart traffic-counter      # needed after editing config.py
```

> `config.py` is read when the script starts, so a change is picked up at the next
> cycle (at most `RUN_MINUTES` later) or right away after a restart.

---

## 7. Dashboard

The dashboard is a Flask app on **port 8080**. Run it **from the project root**,
because `DB_DIR` is a relative path:

```bash
cd ~/Documents/traffic-counter
source traffic/bin/activate
python dashboard/app.py
```

Open `http://traffic-pi.local:8080` (or `http://<pi-ip>:8080`).

| Page / endpoint | Description |
|---|---|
| `/` | Today: total, ↑/↓, split by class, by hour, last 10 passages, sanity check panel |
| `/history` | Several days of history (daily totals, average per hour, totals by class and direction) |
| `GET /api/stats` | Today's statistics as JSON |
| `GET /api/history?days=7` | History over N days |
| `GET /api/sanity/available-logs` | Dates that have a `counter_*.log` |
| `POST /api/sanity/run` | Run the sanity check. Body: `{"date": "YYYY-MM-DD", "min_detections": 5, "use_claude": true, "duplicate_window": 1.0}` |
| `POST /api/sanity/insert` | Insert the missed passages found by the last report |
| `POST /api/sanity/remove-duplicates` | Delete the duplicates found by the last report |
| `GET /api/sanity/report` | Last report (kept in memory) |

### Optional: dashboard as a service

There is no unit file for the dashboard in the repo. Here is one you can use as
`/etc/systemd/system/traffic-dashboard.service` (change the user and paths):

```ini
[Unit]
Description=Traffic Counter dashboard
After=network.target

[Service]
User=abuche
WorkingDirectory=/home/abuche/Documents/traffic-counter
EnvironmentFile=-/home/abuche/Documents/traffic-counter/.env
ExecStart=/home/abuche/Documents/traffic-counter/traffic/bin/python dashboard/app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now traffic-dashboard
```

> The dashboard has no login. Use it only on a trusted local network.

---

## 8. Sanity check

`live_counter.py` writes every YOLO detection to `data/counter_YYYY-MM-DD.log`.
`sanity_check.py` reads that log and:

1. groups the detections into likely vehicle passages (candidates)
2. matches each candidate with the passages stored in the DB
3. reports **missed passages**, **duplicates** (SORT bugs, normal duplicates,
   double inserts) and **slow vehicles**
4. for unclear candidates (low confidence), can ask the **Claude API** for a
   decision. Answers are cached in `data/sanity_cache_YYYY-MM-DD.json`.

### Claude API key (optional)

The `anthropic` client reads the key from the `ANTHROPIC_API_KEY` environment variable:

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env      # .env is in .gitignore
export $(cat .env | xargs)                      # for a manual run
```

The key must be visible to the process that runs the check: your shell for the
CLI, or the dashboard's environment (see `EnvironmentFile=` above). Without a key,
use `--no-claude` (CLI) or `"use_claude": false` (API). Unclear cases are then
skipped instead of being sent to Claude.

### Command line

```bash
python sanity_check.py                         # today, dry run (nothing is changed)
python sanity_check.py --date 2026-10-04       # another day
python sanity_check.py --no-claude             # rules only
python sanity_check.py --insert                # apply: insert missed + delete duplicates
```

| Option | Default | Description |
|---|---|---|
| `--date` | today | Day to check (`YYYY-MM-DD`) |
| `--min-detections` | `5` | Minimum YOLO frames for a real passage |
| `--duplicate-window` | `5.0` | Time window (s) for finding duplicates |
| `--insert` | off | Write the changes to the DB |
| `--dry-run` | off | Force analysis only (wins over `--insert`) |
| `--no-claude` | off | Turn off the Claude API |

Finer thresholds (`MAX_FRAME_GAP_SEC`, `AMBIGUOUS_CONF`, `SLOW_VEHICLE_SEC`, …)
are constants at the top of `sanity_check.py`.

> `--insert` writes the `confidence_level`, `sanity_reason`, `n_frames` and
> `avg_conf` columns. Run `python migrate_add_confidence.py` first if your DBs
> don't have them yet (see [9](#9-data-and-database-schema)).

---

## 9. Data and database schema

Everything goes into `data/`:

| File | Content |
|---|---|
| `traffic_YYYY-MM-DD.db` | One SQLite DB per day, table `passages` |
| `counter_YYYY-MM-DD.log` | Detection log (only `YOLO det`, `FRANCHISSEMENT`, `PASSAGE ENREGISTRE` lines). The sanity check reads it. |
| `runner.log` | Cycle log from `run_counter.sh` |
| `sanity_cache_YYYY-MM-DD.json` | Cache of Claude decisions |
| `roi.json` | Optional ROI polygon |

Table `passages` as created by `live_counter.py`:

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `timestamp` | TEXT | `YYYY-MM-DD HH:MM:SS` |
| `class` | TEXT | `car`, `truck`, `bus`, `motorcycle`, `bicycle` (`unknown` at night) |
| `direction` | TEXT | `↑` / `↓` |
| `track_id` | INTEGER | SORT track id (`-1` for night detections) |
| `bbox_w`, `bbox_h` | INTEGER | Bbox size at the crossing |

Columns added by `migrate_add_confidence.py` (used by the sanity check):
`confidence_level` (`auto`, `high`, `medium`, `low`, `manual`), `sanity_reason`,
`n_frames`, `avg_conf`.

Migrations run on every `data/traffic_*.db` and can be run again safely:

```bash
python migrate_add_bbox.py
python migrate_add_confidence.py
```

> When a new daily DB is created, it **does not** have the `migrate_add_confidence.py`
> columns. Run the migration again before inserting with the sanity check.

Quick look at the data:

```bash
sqlite3 data/traffic_$(date +%F).db \
  "SELECT class, direction, COUNT(*) FROM passages GROUP BY 1,2;"
```

---

## 10. Other tools

```bash
# Offline counting on a video (writes output_counted.mp4 and ./traffic.db)
python count_phase2.py videos/IMG_4835.MOV --output out.mp4 --line-y 215 --conf 0.4
python count_phase2.py raw.mp4 --rotate     # rotate 90° clockwise first if the video is not oriented yet

# Inference benchmark
python benchmark.py
```

---

## 11. Known issues and caveats

These come from reading the current code. They are listed here so they don't
surprise you during setup.

- **Night mode crashes.** `night_detector.py` uses `config.COUNT_LINE_X`, which does not
  exist in `config.py`. It also looks for a **vertical** line crossing (`→`/`←`),
  while the day mode counts on a horizontal line. When the brightness drops below
  `NIGHT_BRIGHTNESS_THRESHOLD`, the processing thread fails with an
  `AttributeError` and nothing is counted until the next cycle. Until this is fixed,
  you can set `NIGHT_BRIGHTNESS_THRESHOLD = 0` to keep YOLO running all the time.
- **The day reset doesn't switch DB files.** At midnight `live_counter.py` clears its
  track state, but it keeps writing to the DB file it opened at startup. The
  periodic restart by `run_counter.sh` opens the new day's file, so passages
  between midnight and the next restart are stored in the previous day's DB.
- **The service user doesn't match the paths:** `User=pi` vs `/home/abuche/...`
  (see 3.8).
- **The `run_counter.sh` comments say 59 min**, but `RUN_MINUTES=10`.
- `calibrate_brightness.py` sets up the camera with the post-rotation size instead
  of the native one. The brightness reading is still fine.
- `dashboard/config.py` and `dashboard/sanity_check.py` are old copies. The
  dashboard uses the files at the project root.
- `nohup.out` and `dashboard/.DS_Store` are committed by mistake.

---

## 12. Troubleshooting

| Problem | What to check |
|---|---|
| `ModuleNotFoundError: picamera2` in the venv | The venv was created without `--system-site-packages`. Create it again (3.5). |
| `ModuleNotFoundError: sort` | `sort/` is missing. Clone it (3.6). |
| Error about `TkAgg` / `tkinter` when importing SORT | Edit `sort/sort.py` and use `matplotlib.use('Agg')` |
| `yolov8n_ncnn_model` not found | Export the model (3.7) and run from the project root |
| `FileNotFoundError` on `data/counter_....log` | Create the `data/` folder and run from the project root |
| Camera busy / `Device or resource busy` | Only one process can use the camera. Stop the service before `check_camera.py` or `calibrate_brightness.py`. |
| `WATCHDOG — aucune frame depuis Ns` | Camera cable, power, or overheating (`vcgencmd measure_temp`) |
| Vehicles seen but not counted | Check `COUNT_LINE_Y` with `check_camera.py`, make `COUNT_LINE_MARGIN` larger, lower `SORT_MIN_HITS` to 1, check the ROI |
| Double counts | Raise `COUNTING_COOLDOWN_*` and/or `SORT_MIN_HITS`, then run the sanity check to clean up |
| Dashboard shows 0 | It must be started from the project root. Check that `data/traffic_<today>.db` exists. |
| Sanity check says "Erreur API" | `ANTHROPIC_API_KEY` is not set for that process. Use `--no-claude` or set the key. |
