import sqlite3
from pathlib import Path
from datetime import date, timedelta
import config

def migrate(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    # Vérifie si la colonne existe déjà
    cols = [r[1] for r in conn.execute("PRAGMA table_info(passages)").fetchall()]
    if "bbox_w" not in cols:
        conn.execute("ALTER TABLE passages ADD COLUMN bbox_w INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE passages ADD COLUMN bbox_h INTEGER DEFAULT 0")
        conn.commit()
        print(f"  Migré : {db_path.name}")
    else:
        print(f"  Déjà à jour : {db_path.name}")
    conn.close()

# Migre tous les fichiers DB existants
data_dir = Path(config.DB_DIR)
for db_path in sorted(data_dir.glob("traffic_*.db")):
    migrate(db_path)
print("Migration terminée.")