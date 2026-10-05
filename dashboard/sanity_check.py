# sanity_check.py
import re
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
# import anthropic
import config

log = logging.getLogger("sanity")

# ── Paramètres ─────────────────────────────────────────────────────────────────
MIN_DETECTIONS      = 5      # minimum de frames YOLO pour un vrai passage
DB_MATCH_WINDOW_SEC = 1.0    # fenêtre de tolérance pour matcher en DB (secondes)
MAX_FRAME_GAP_SEC   = 2.0    # gap max entre deux frames d'une même séquence
MIN_BBOX_OVERLAP    = 50     # distance max entre centres de bbox consécutives (px)
AMBIGUOUS_CONF      = 0.35   # conf moyenne sous ce seuil → analyse Claude

# ── Structures de données ──────────────────────────────────────────────────────
@dataclass
class Detection:
    timestamp: datetime
    cls:       str
    conf:      float
    cx:        float   # centre X
    cy:        float   # centre Y

@dataclass
class Candidate:
    detections:  list[Detection] = field(default_factory=list)
    cls:         str = ""
    direction:   str = ""
    cross_time:  Optional[datetime] = None
    avg_conf:    float = 0.0
    is_ambiguous: bool = False

    @property
    def start_time(self) -> Optional[datetime]:
        return self.detections[0].timestamp if self.detections else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.detections[-1].timestamp if self.detections else None

# ── Parsing des logs ───────────────────────────────────────────────────────────
LOG_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+INFO\s+"
    r"YOLO det\s+(\w+)\s+conf=([\d.]+)\s+bbox=\((\d+),(\d+),(\d+),(\d+)\)"
)

def parse_log(log_path: Path) -> list[Detection]:
    detections = []
    with open(log_path, "r") as f:
        for line in f:
            m = LOG_PATTERN.search(line)
            if not m:
                continue
            ts_str, cls, conf, x1, y1, x2, y2 = m.groups()
            ts  = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            cx  = (int(x1) + int(x2)) / 2
            cy  = (int(y1) + int(y2)) / 2
            detections.append(Detection(
                timestamp=ts,
                cls=cls,
                conf=float(conf),
                cx=cx,
                cy=cy
            ))
    log.info(f"Parse : {len(detections)} détections YOLO extraites")
    return detections

# ── Clustering en séquences ────────────────────────────────────────────────────
def cluster_detections(detections: list[Detection],
                       min_dets: int = MIN_DETECTIONS) -> list[Candidate]:
    """
    Regroupe les détections individuelles en séquences cohérentes
    représentant un seul passage de véhicule.
    """
    if not detections:
        return []

    candidates = []
    current    = [detections[0]]

    for det in detections[1:]:
        prev = current[-1]
        dt   = (det.timestamp - prev.timestamp).total_seconds()
        dist = ((det.cx - prev.cx)**2 + (det.cy - prev.cy)**2) ** 0.5

        # Même séquence si : gap temporel court ET déplacement spatial cohérent
        if dt <= MAX_FRAME_GAP_SEC and dist <= MIN_BBOX_OVERLAP:
            current.append(det)
        else:
            if len(current) >= min_dets:
                candidates.append(_build_candidate(current))
            current = [det]

    if len(current) >= min_dets:
        candidates.append(_build_candidate(current))

    log.info(f"Clustering : {len(candidates)} candidats passages identifiés")
    return candidates

def _build_candidate(dets: list[Detection]) -> Candidate:
    """Construit un Candidate depuis une séquence de détections."""
    cls_counts = defaultdict(int)
    for d in dets:
        cls_counts[d.cls] += 1
    dominant_cls = max(cls_counts, key=cls_counts.get)

    # Direction via déplacement Y moyen
    dy = dets[-1].cy - dets[0].cy
    direction = "↓" if dy > 0 else "↑"

    # Timestamp estimé du franchissement — frame la plus proche de COUNT_LINE_Y
    line_y    = config.COUNT_LINE_Y
    margin    = getattr(config, "COUNT_LINE_MARGIN", 20)
    cross_det = min(dets, key=lambda d: abs(d.cy - line_y))
    crosses_line = abs(cross_det.cy - line_y) <= (margin + 30)

    avg_conf    = sum(d.conf for d in dets) / len(dets)
    is_ambiguous = (
        avg_conf < AMBIGUOUS_CONF or
        not crosses_line or
        len(dets) < MIN_DETECTIONS + 2
    )

    return Candidate(
        detections=dets,
        cls=dominant_cls,
        direction=direction,
        cross_time=cross_det.timestamp if crosses_line else None,
        avg_conf=avg_conf,
        is_ambiguous=is_ambiguous
    )

# ── Matching avec la DB ────────────────────────────────────────────────────────
def load_db_passages(db_path: Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM passages ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def match_candidates(candidates: list[Candidate],
                     db_passages: list[dict]) -> tuple[list, list]:
    """
    Retourne (counted, missed) — candidats matchés et non matchés en DB.
    """
    counted = []
    missed  = []
    window  = timedelta(seconds=DB_MATCH_WINDOW_SEC)

    for cand in candidates:
        if not cand.cross_time:
            continue   # ne traverse pas la ligne — pas un passage

        matched = False
        for passage in db_passages:
            db_ts = datetime.strptime(passage["timestamp"], "%Y-%m-%d %H:%M:%S")
            if (abs((cand.cross_time - db_ts).total_seconds()) <= DB_MATCH_WINDOW_SEC
                    and passage["direction"] == cand.direction):
                matched = True
                break

        if matched:
            counted.append(cand)
        else:
            missed.append(cand)

    log.info(
        f"Matching : {len(counted)} comptés, {len(missed)} manqués"
    )
    return counted, missed

# ── Analyse Claude pour cas ambigus ───────────────────────────────────────────
def analyze_with_claude(candidate: Candidate) -> dict:
    """
    Envoie un candidat ambigu à Claude API pour décision.
    Retourne {"should_count": bool, "reason": str, "direction": str, "cls": str}
    """
    client = anthropic.Anthropic()

    det_summary = "\n".join([
        f"  t={d.timestamp.strftime('%H:%M:%S')}  "
        f"cls={d.cls}  conf={d.conf:.2f}  "
        f"cx={d.cx:.0f}  cy={d.cy:.0f}"
        for d in candidate.detections
    ])

    prompt = f"""Tu analyses un système de comptage de trafic routier.
    Voici une séquence de détections YOLO pour un potentiel passage de véhicule :

    {det_summary}

    Paramètres du système :
    - Ligne de comptage Y = {config.COUNT_LINE_Y} (±{getattr(config, 'COUNT_LINE_MARGIN', 20)}px)
    - Direction ↑ = Y décroissant (véhicule s'approche)
    - Direction ↓ = Y croissant (véhicule s'éloigne)
    - Nombre de détections : {len(candidate.detections)}
    - Confiance moyenne : {candidate.avg_conf:.2f}
    - Direction estimée : {candidate.direction}

    Question : ce véhicule a-t-il réellement traversé la ligne de comptage ?
    Réponds UNIQUEMENT en JSON valide :
    {{"should_count": true/false, "reason": "explication courte", "direction": "↑ ou ↓", "cls": "car/truck/bicycle/etc"}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        import json
        text = response.content[0].text.strip()
        return json.loads(text)
    except Exception as e:
        log.error(f"Erreur Claude API : {e}")
        return {
            "should_count": False,
            "reason": f"Erreur API : {e}",
            "direction": candidate.direction,
            "cls": candidate.cls
        }

# ── Insertion en DB ────────────────────────────────────────────────────────────
def insert_missed(db_path: Path, candidate: Candidate,
                  cls: str, direction: str) -> bool:
    if not candidate.cross_time:
        return False
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO passages (timestamp, class, direction, track_id) "
            "VALUES (?, ?, ?, ?)",
            (
                candidate.cross_time.strftime("%Y-%m-%d %H:%M:%S"),
                cls,
                direction,
                -99   # track_id spécial = inséré manuellement par sanity check
            )
        )
        conn.commit()
        conn.close()
        log.info(
            f"Inséré manuellement : {cls} {direction} "
            f"@ {candidate.cross_time}"
        )
        return True
    except Exception as e:
        log.error(f"Erreur insertion : {e}")
        return False

# ── Point d'entrée principal ───────────────────────────────────────────────────
def run_sanity_check(target_date: date = None,
                     min_detections: int = MIN_DETECTIONS) -> dict:
    """
    Lance le sanity check complet pour une date donnée.
    Retourne un rapport structuré.
    """
    if target_date is None:
        target_date = date.today()

    log_path = Path(config.DB_DIR) / f"counter_{target_date.isoformat()}.log"
    db_path  = Path(config.DB_DIR) / f"traffic_{target_date.isoformat()}.db"

    if not log_path.exists():
        return {"error": f"Log introuvable : {log_path}"}

    # Phase 1 — Parse
    detections = parse_log(log_path)
    if not detections:
        return {"error": "Aucune détection YOLO dans les logs"}

    # Phase 2 — Clustering
    candidates = cluster_detections(detections, min_dets=min_detections)

    # Phase 3 — Matching DB
    db_passages = load_db_passages(db_path)
    counted, missed = match_candidates(candidates, db_passages)

    # Phase 4 — Analyse des manqués
    results_missed = []
    for cand in missed:
        if cand.is_ambiguous:
            # Cas ambigu → Claude API
            # claude_result = analyze_with_claude(cand)
            results_missed.append({
                "timestamp":    cand.cross_time.isoformat() if cand.cross_time else None,
                "cls":          "N/A", #claude_result["cls"],
                "direction":    "N/A", #claude_result["direction"],
                "avg_conf":     round(cand.avg_conf, 2),
                "n_frames":     len(cand.detections),
                "should_count": "N/A", #claude_result["should_count"],
                "reason":       "N/A", #claude_result["reason"],
                "method":       "claude",
                "inserted":     False,
            })
        else:
            # Cas clair → règles fixes
            results_missed.append({
                "timestamp":    cand.cross_time.isoformat() if cand.cross_time else None,
                "cls":          cand.cls,
                "direction":    cand.direction,
                "avg_conf":     round(cand.avg_conf, 2),
                "n_frames":     len(cand.detections),
                "should_count": True,
                "reason":       f"Séquence claire ({len(cand.detections)} frames, conf={cand.avg_conf:.2f})",
                "method":       "rules",
                "inserted":     False,
            })

    return {
        "date":           target_date.isoformat(),
        "log_path":       str(log_path),
        "db_path":        str(db_path),
        "total_yolo":     len(detections),
        "total_candidates": len(candidates),
        "total_counted":  len(counted) + len(db_passages),
        "db_passages":    len(db_passages),
        "missed":         results_missed,
        "missed_count":   len([m for m in results_missed if m["should_count"]]),
    }

def insert_all_missed(report: dict) -> dict:
    """Insère tous les passages manqués validés dans la DB."""
    db_path = Path(report["db_path"])
    inserted = 0
    for item in report["missed"]:
        if item["should_count"] and not item["inserted"]:
            ts  = datetime.fromisoformat(item["timestamp"])
            cand = Candidate(
                cross_time=ts,
                cls=item["cls"],
                direction=item["direction"]
            )
            if insert_missed(db_path, cand, item["cls"], item["direction"]):
                item["inserted"] = True
                inserted += 1
    report["inserted_count"] = inserted
    return report