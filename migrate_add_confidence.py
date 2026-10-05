import sqlite3
from pathlib import Path
import config

data_dir = Path(config.DB_DIR)
for db_path in sorted(data_dir.glob("traffic_*.db")):
    conn = sqlite3.connect(str(db_path))
    cols = [r[1] for r in conn.execute("PRAGMA table_info(passages)").fetchall()]

    if "confidence_level" not in cols:
        conn.execute("""
            ALTER TABLE passages
            ADD COLUMN confidence_level TEXT DEFAULT 'auto'
        """)
        # auto     = passage détecté par live_counter (track_id >= 0)
        # high     = inséré par sanity check, règles claires + conf YOLO > 0.4
        # medium   = inséré par sanity check, règles claires + conf YOLO 0.25-0.4
        # low      = inséré par sanity check, cas ambigu validé par Claude
        # manual   = inséré à la main par l'utilisateur

    if "sanity_reason" not in cols:
        conn.execute("""
            ALTER TABLE passages
            ADD COLUMN sanity_reason TEXT DEFAULT NULL
        """)
        # Stocke la raison textuelle du sanity check

    if "n_frames" not in cols:
        conn.execute("""
            ALTER TABLE passages
            ADD COLUMN n_frames INTEGER DEFAULT 0
        """)

    if "avg_conf" not in cols:
        conn.execute("""
            ALTER TABLE passages
            ADD COLUMN avg_conf REAL DEFAULT 0.0
        """)

    conn.commit()
    conn.close()
    print(f"Migré : {db_path.name}")

print("Migration terminée.")