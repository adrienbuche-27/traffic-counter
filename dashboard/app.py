from flask import Flask, jsonify, render_template, request
import sqlite3
from pathlib import Path
from datetime import datetime, date
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

app = Flask(__name__)

sys.path.insert(0, "/home/abuche/traffic-counter")
from sanity_check import run_sanity_check, insert_all_missed, remove_all_duplicates
import json

# Cache du dernier rapport en mémoire
_last_report = {}

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/api/history")
def history_stats():
    from datetime import timedelta

    # Paramètres période
    days    = int(request.args.get("days", 7))
    end     = date.today()
    start   = end - timedelta(days=days - 1)

    # Collecte des données sur tous les fichiers DB de la période
    daily_data  = []
    hour_totals = defaultdict(int)
    hour_counts = defaultdict(int)

    current = start
    while current <= end:
        db_path = Path(config.DB_DIR) / f"traffic_{current.isoformat()}.db"

        if db_path.exists():
            rows = query_db(db_path, "SELECT * FROM passages")

            # Comptage par classe et direction
            by_class = defaultdict(int)
            by_dir   = defaultdict(int)
            for r in rows:
                by_class[r["class"]]     += 1
                by_dir[r["direction"]]   += 1

                # Accumule pour la moyenne horaire
                h = r["timestamp"][11:13]   # "HH"
                hour_totals[h] += 1
                hour_counts[h]  = hour_counts.get(h, 0)

            # Accumule le count horaire par jour
            for r in rows:
                h = r["timestamp"][11:13]
                hour_counts[h] = days  # diviseur pour la moyenne

            daily_data.append({
                "date":      current.isoformat(),
                "total":     len(rows),
                "by_class":  dict(by_class),
                "by_dir":    dict(by_dir),
            })
        else:
            daily_data.append({
                "date":     current.isoformat(),
                "total":    0,
                "by_class": {},
                "by_dir":   {},
            })

        current += timedelta(days=1)

    # Moyenne horaire sur la période
    by_hour_avg = [
        {
            "hour": f"{h:02d}h",
            "avg":  round(hour_totals.get(f"{h:02d}", 0) / days, 1)
        }
        for h in range(24)
    ]

    # Totaux globaux sur la période
    all_rows     = []
    total_by_class = defaultdict(int)
    total_by_dir   = defaultdict(int)
    for d in daily_data:
        for cls, n in d["by_class"].items():
            total_by_class[cls] += n
        for dr, n in d["by_dir"].items():
            total_by_dir[dr] += n

    return jsonify({
        "start":          start.isoformat(),
        "end":            end.isoformat(),
        "days":           days,
        "daily":          daily_data,
        "by_hour_avg":    by_hour_avg,
        "total_by_class": dict(total_by_class),
        "total_by_dir":   dict(total_by_dir),
        "grand_total":    sum(d["total"] for d in daily_data),
        "updated_at":     datetime.now().strftime("%H:%M:%S"),
    })

def query_db(db_path, sql, params=()):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.route("/api/sanity/available-logs")
def sanity_available_logs():
    data_dir = Path(config.DB_DIR)
    logs = sorted(
        [f.stem.replace("counter_", "")
         for f in data_dir.glob("counter_*.log")],
        reverse=True
    )
    return jsonify({"logs": logs})

@app.route("/api/sanity/remove-duplicates", methods=["POST"])
def sanity_remove_duplicates():
    global _last_report
    if not _last_report:
        return jsonify({"error": "Lance d'abord le sanity check"}), 400
    if not _last_report.get("duplicates"):
        return jsonify({"error": "Aucun doublon détecté dans le dernier rapport"}), 400
    _last_report = remove_all_duplicates(_last_report)
    return jsonify(_last_report)


@app.route("/api/sanity/run", methods=["POST"])
def sanity_run():
    """Lance le sanity check — peut prendre 30-60s si Claude API appelé."""
    data       = request.get_json() or {}
    min_dets   = int(data.get("min_detections", 5))
    target_str = data.get("date", date.today().isoformat())
    target     = date.fromisoformat(target_str)
    use_claude = bool(data.get("use_claude", True))   # ← nouveau paramètre
    duplicate_window = float(data.get("duplicate_window", 1.0))   # ← nouveau

    global _last_report
    _last_report = run_sanity_check(target_date=target, min_detections=min_dets, use_claude=use_claude, duplicate_window=duplicate_window)
    return jsonify(_last_report)

@app.route("/api/sanity/insert", methods=["POST"])
def sanity_insert():
    """Insère les passages manqués validés en DB."""
    global _last_report
    if not _last_report:
        return jsonify({"error": "Lance d'abord le sanity check"}), 400
    _last_report = insert_all_missed(_last_report)
    return jsonify(_last_report)

@app.route("/api/sanity/report")
def sanity_report():
    return jsonify(_last_report)

def get_db_path() -> Path:
    return Path(config.DB_DIR) / f"traffic_{date.today().isoformat()}.db"

def query(sql, params=()):
    db_path = get_db_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def stats():
    # Total du jour
    result = query("SELECT COUNT(*) as n FROM passages")
    total  = result[0]["n"] if result else 0

    # Répartition par classe
    by_class = query("""
        SELECT class, COUNT(*) as n
        FROM passages GROUP BY class ORDER BY n DESC
    """)

    # Répartition par direction — ↑ et ↓
    by_direction = query("""
        SELECT direction, COUNT(*) as n
        FROM passages GROUP BY direction
    """)

    # Passages par heure
    by_hour_raw = query("""
        SELECT strftime('%H', timestamp) as hour, COUNT(*) as n
        FROM passages GROUP BY hour ORDER BY hour
    """)

    hour_map = {row["hour"]: row["n"] for row in by_hour_raw}
    current_hour = datetime.now().hour
    by_hour = [
        {"hour": f"{h:02d}h", "n": hour_map.get(f"{h:02d}", 0)}
        for h in range(current_hour + 1)
    ]

    # Derniers passages
    recent = query("""
        SELECT timestamp, class, direction
        FROM passages ORDER BY timestamp DESC LIMIT 10
    """)

    # Comptage ↑ et ↓ séparément pour les KPIs
    up   = next((d["n"] for d in by_direction if d["direction"] == "↑"), 0)
    down = next((d["n"] for d in by_direction if d["direction"] == "↓"), 0)

    return jsonify({
        "total":        total,
        "up":           up,
        "down":         down,
        "by_class":     by_class,
        "by_direction": by_direction,
        "by_hour":      by_hour,
        "recent":       recent,
        "updated_at":   datetime.now().strftime("%H:%M:%S"),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)