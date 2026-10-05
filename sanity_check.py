# sanity_check.py
import re
import sys
import sqlite3
import logging
import anthropic
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import config
import json
import hashlib


log = logging.getLogger("sanity")

# ── Paramètres ─────────────────────────────────────────────────────────────────
MIN_DETECTIONS       = 5      # minimum de frames YOLO pour un vrai passage
DB_MATCH_WINDOW_SEC  = 1.0    # fenêtre de tolérance pour matcher en DB
MAX_FRAME_GAP_SEC    = 2.0    # gap max entre deux frames d'une même séquence
MIN_BBOX_OVERLAP     = 50     # distance max entre centres de bbox consécutives (px)
AMBIGUOUS_CONF       = 0.35   # conf moyenne sous ce seuil → analyse Claude
SPATIAL_SPLIT_DX     = 80     # distance X min pour forcer une coupure de séquence
SORT_DUPLICATE_SEC   = 0.5    # delta < N secondes + bbox identique = bug SORT
SORT_DUPLICATE_PX    = 5      # distance bbox < N px = bug SORT
SLOW_VEHICLE_SEC     = 15     # présence > N secondes = véhicule lent à signaler

# ── Structures de données ──────────────────────────────────────────────────────
@dataclass
class Detection:
    timestamp: datetime
    cls:       str
    conf:      float
    cx:        float
    cy:        float

@dataclass
class Candidate:
    detections:   list = field(default_factory=list)
    cls:          str = ""
    direction:    str = ""
    cross_time:   Optional[datetime] = None
    avg_conf:     float = 0.0
    is_ambiguous: bool = False
    note:         str = ""
    est_bbox_w:   float = 0.0
    est_bbox_h:   float = 0.0

    @property
    def start_time(self) -> Optional[datetime]:
        return self.detections[0].timestamp if self.detections else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.detections[-1].timestamp if self.detections else None

    @property
    def duration_sec(self) -> float:
        if not self.detections or len(self.detections) < 2:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()

# ── Parsing des logs ───────────────────────────────────────────────────────────
LOG_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+INFO\s+"
    r"YOLO det\s+(\w+)\s+conf=([\d.]+)\s+bbox=\((\d+),(\d+),(\d+),(\d+)\)"
)

def parse_log(log_path: Path) -> list:
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
                timestamp=ts, cls=cls, conf=float(conf), cx=cx, cy=cy
            ))
    log.info(f"Parse : {len(detections)} détections YOLO extraites")
    return detections

# ── Clustering en séquences ────────────────────────────────────────────────────
def cluster_detections(detections: list, min_dets: int = MIN_DETECTIONS) -> list:
    """
    Regroupe les détections en séquences cohérentes.
    Amélioration 1 : coupe forcée si distance X > SPATIAL_SPLIT_DX
    (deux objets distincts dans le champ simultanément).
    """
    if not detections:
        return []

    candidates = []
    current    = [detections[0]]

    for det in detections[1:]:
        prev = current[-1]
        dt   = (det.timestamp - prev.timestamp).total_seconds()
        dist = ((det.cx - prev.cx)**2 + (det.cy - prev.cy)**2) ** 0.5
        dx   = abs(det.cx - prev.cx)

        should_cut = (
            dt > MAX_FRAME_GAP_SEC or
            dist > MIN_BBOX_OVERLAP or
            (dx > SPATIAL_SPLIT_DX and dt < 1.0)
        )

        if should_cut:
            if len(current) >= min_dets:
                candidates.append(_build_candidate(current))
            current = [det]
        else:
            current.append(det)

    if len(current) >= min_dets:
        candidates.append(_build_candidate(current))

    log.info(f"Clustering : {len(candidates)} candidats passages identifiés")
    return candidates

def _build_candidate(dets: list) -> Candidate:
    cls_counts = defaultdict(int)
    for d in dets:
        cls_counts[d.cls] += 1
    dominant_cls = max(cls_counts, key=cls_counts.get)

    dy        = dets[-1].cy - dets[0].cy
    direction = "↓" if dy > 0 else "↑"

    line_y    = config.COUNT_LINE_Y
    margin    = getattr(config, "COUNT_LINE_MARGIN", 20)
    cross_det = min(dets, key=lambda d: abs(d.cy - line_y))
    crosses_line = abs(cross_det.cy - line_y) <= (margin + 30)

    avg_conf     = sum(d.conf for d in dets) / len(dets)
    duration     = (dets[-1].timestamp - dets[0].timestamp).total_seconds()

    is_ambiguous = (
        avg_conf < AMBIGUOUS_CONF or
        not crosses_line or
        len(dets) < MIN_DETECTIONS + 2
    )

    note = ""
    if duration > SLOW_VEHICLE_SEC:
        note = (
            f"Véhicule lent ({duration:.0f}s dans le champ) — "
            f"vérifier si comptage correct"
        )

    cross_dets = [d for d in dets
                  if abs(d.cy - config.COUNT_LINE_Y) <= (margin + 30)]
    if cross_dets:
        est_bbox_w = max(d.cx for d in cross_dets) - min(d.cx for d in cross_dets)
        est_bbox_h = max(d.cy for d in cross_dets) - min(d.cy for d in cross_dets)
    else:
        est_bbox_w = 0.0
        est_bbox_h = 0.0

    return Candidate(
        detections=dets,
        cls=dominant_cls,
        direction=direction,
        cross_time=cross_det.timestamp if crosses_line else None,
        avg_conf=avg_conf,
        is_ambiguous=is_ambiguous,
        note=note,
        est_bbox_w=round(est_bbox_w, 1),
        est_bbox_h=round(est_bbox_h, 1)
    )

# ── Matching avec la DB ────────────────────────────────────────────────────────
def load_db_passages(db_path: Path) -> list:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM passages ORDER BY timestamp"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def match_candidates(candidates: list,
                     db_passages: list) -> tuple:
    """
    Matching candidats log → passages DB.
    Utilise uniquement les passages fiables (auto ou manuel high/medium).
    Les passages 'low' ne bloquent pas la re-détection.
    Vérifie aussi la cohérence des bboxes si disponibles en DB.
    """
    counted = []
    missed  = []

    # Passages fiables uniquement — les "low" ne bloquent pas la re-détection
    reliable_passages = [
        p for p in db_passages
        if p.get("track_id", 0) >= 0
        or p.get("confidence_level") in ("high", "medium")
        or (p.get("track_id") == -99                               # manuel sans niveau
        and p.get("confidence_level") is None)
    ]

    for cand in candidates:
        if not cand.cross_time:
            continue

        matched = False
        for passage in reliable_passages:
            db_ts = datetime.strptime(passage["timestamp"], "%Y-%m-%d %H:%M:%S")
            dt    = abs((cand.cross_time - db_ts).total_seconds())

            if dt > DB_MATCH_WINDOW_SEC:
                continue
            if passage["direction"] != cand.direction:
                continue

            # # ── Vérification bbox si disponible en DB ─────────────────────────
            # db_w = passage.get("bbox_w", 0)
            # db_h = passage.get("bbox_h", 0)

            # if db_w > 0 and db_h > 0 and cand.detections:
            #     cand_w = abs(
            #         max(d.cx for d in cand.detections) -
            #         min(d.cx for d in cand.detections)
            #     )
            #     cand_h = abs(
            #         max(d.cy for d in cand.detections) -
            #         min(d.cy for d in cand.detections)
            #     )

            #     if cand_w > 0 and cand_h > 0:
            #         area_db   = db_w   * db_h
            #         area_cand = cand_w * cand_h
            #         ratio = min(area_db, area_cand) / max(area_db, area_cand)

            #         if ratio < getattr(config, "BBOX_SIMILARITY_RATIO", 0.5):
            #             log.debug(
            #                 f"Match rejeté — bbox incohérentes "
            #                 f"DB={db_w}x{db_h} vs candidat≈{cand_w:.0f}x{cand_h:.0f} "
            #                 f"ratio={ratio:.2f}"
            #             )
            #             continue

            matched = True
            break

        if matched:
            counted.append(cand)
        else:
            missed.append(cand)

    log.info(
        f"Matching sur {len(reliable_passages)}/{len(db_passages)} passages fiables "
        f"— {len(counted)} matchés, {len(missed)} manqués"
    )
    return counted, missed

# ── Détection des doublons en DB ───────────────────────────────────────────────
def find_db_duplicates(db_passages: list, window_sec: float = 5.0) -> list:
    """
    Détecte les doublons en DB.
    - Ignore les paires auto+manuel (le manuel complète le manquant)
    - Bug SORT : deux passages automatiques avec delta < SORT_DUPLICATE_SEC
    - Double insertion manuelle : deux passages -99 très proches
    - Doublon standard : deux passages automatiques dans la fenêtre temporelle
    - Vérifie la cohérence des bboxes pour confirmer ou infirmer
    """
    duplicates = []
    checked    = set()

    for i, p1 in enumerate(db_passages):
        if i in checked:
            continue
        ts1         = datetime.strptime(p1["timestamp"], "%Y-%m-%d %H:%M:%S")
        is_manual_1 = p1.get("track_id") == -99

        for j, p2 in enumerate(db_passages):
            if j <= i or j in checked:
                continue
            ts2         = datetime.strptime(p2["timestamp"], "%Y-%m-%d %H:%M:%S")
            is_manual_2 = p2.get("track_id") == -99

            dt       = abs((ts2 - ts1).total_seconds())
            same_dir = p1["direction"] == p2["direction"]
            same_cls = p1["class"]     == p2["class"]

            if not (same_dir and same_cls):
                continue

            # Paire auto+manuel → le manuel complète le manquant → jamais un doublon
            if (not is_manual_1 and is_manual_2) or \
               (is_manual_1 and not is_manual_2):
                log.debug(
                    f"Paire auto+manuel ignorée — "
                    f"id={p1['id']} (track={p1.get('track_id')}) + "
                    f"id={p2['id']} (track={p2.get('track_id')}) "
                    f"dt={dt:.1f}s"
                )
                continue

            # ── Analyse bbox si disponible ────────────────────────────────────
            bbox_similar = True
            bbox_note    = ""

            w1 = p1.get("bbox_w", 0)
            h1 = p1.get("bbox_h", 0)
            w2 = p2.get("bbox_w", 0)
            h2 = p2.get("bbox_h", 0)

            if w1 > 0 and h1 > 0 and w2 > 0 and h2 > 0:
                area1 = w1 * h1
                area2 = w2 * h2
                ratio = min(area1, area2) / max(area1, area2)
                bbox_similar = ratio >= getattr(config, "BBOX_SIMILARITY_RATIO", 0.5)
                bbox_note = (
                    f"bbox similaires (ratio={ratio:.2f})"
                    if bbox_similar else
                    f"bbox différentes (ratio={ratio:.2f} < {getattr(config, 'BBOX_SIMILARITY_RATIO', 0.5)})"
                )

            # ── Bug SORT : deux passages automatiques très proches ────────────
            if dt <= SORT_DUPLICATE_SEC and not is_manual_1 and not is_manual_2:
                duplicates.append({
                    "keep":       p1,
                    "remove":     p2,
                    "delta_sec":  round(dt, 1),
                    "type":       "sort_bug",
                    "reason":     f"Bug SORT ({dt:.1f}s) {bbox_note}",
                    "bbox_check": bbox_similar,
                    "removed":    False,
                })
                checked.add(j)

            # ── Double insertion manuelle ─────────────────────────────────────
            elif dt <= SORT_DUPLICATE_SEC and is_manual_1 and is_manual_2:
                duplicates.append({
                    "keep":       p1,
                    "remove":     p2,
                    "delta_sec":  round(dt, 1),
                    "type":       "manual_dup",
                    "reason":     f"Double insertion sanity check ({dt:.1f}s)",
                    "bbox_check": bbox_similar,
                    "removed":    False,
                })
                checked.add(j)

            # ── Doublon standard : deux passages automatiques uniquement ───────
            elif dt <= window_sec and not is_manual_1 and not is_manual_2:
                if bbox_similar:
                    duplicates.append({
                        "keep":       p1,
                        "remove":     p2,
                        "delta_sec":  round(dt, 1),
                        "type":       "standard",
                        "reason":     f"Double comptage ({dt:.1f}s) — {bbox_note}",
                        "bbox_check": True,
                        "removed":    False,
                    })
                    checked.add(j)
                else:
                    log.debug(
                        f"Faux doublon ignoré : {p1['class']} {p1['direction']} "
                        f"dt={dt:.1f}s mais {bbox_note}"
                    )

    sort_bugs   = sum(1 for d in duplicates if d["type"] == "sort_bug")
    standard    = sum(1 for d in duplicates if d["type"] == "standard")
    manual_dups = sum(1 for d in duplicates if d["type"] == "manual_dup")
    log.info(
        f"Doublons DB : {len(duplicates)} total "
        f"({sort_bugs} bugs SORT, {standard} standard, {manual_dups} double-insertion)"
    )
    return duplicates

def remove_db_duplicates(db_path: Path, duplicates: list) -> int:
    if not duplicates:
        return 0
    conn    = sqlite3.connect(str(db_path))
    removed = 0
    for dup in duplicates:
        conn.execute("DELETE FROM passages WHERE id = ?", (dup["remove"]["id"],))
        removed += 1
        log.info(
            f"Supprimé [{dup['type']}] id={dup['remove']['id']}  "
            f"{dup['remove']['class']} {dup['remove']['direction']}  "
            f"@ {dup['remove']['timestamp']}  ({dup['reason']})"
        )
    conn.commit()
    conn.close()
    return removed

# ── Analyse Claude pour cas ambigus ───────────────────────────────────────────
def analyze_with_claude(candidate: Candidate,
                        cache: dict,
                        target_date: date) -> dict:
    key = _candidate_key(candidate)

    if key in cache:
        cached = cache[key]
        log.info(f"Cache HIT — clé {key} ({cached['cls']} {cached['direction']})")
        return {**cached, "from_cache": True}

    log.info(f"Cache MISS — appel Claude API pour clé {key}")
    client = anthropic.Anthropic()

    det_summary = "\n".join([
        f"  t={d.timestamp.strftime('%H:%M:%S')}  "
        f"cls={d.cls}  conf={d.conf:.2f}  cx={d.cx:.0f}  cy={d.cy:.0f}"
        for d in candidate.detections
    ])

    slow_note = f"\nNote : {candidate.note}" if candidate.note else ""

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
- Durée dans le champ : {candidate.duration_sec:.1f}s{slow_note}
- Bbox estimée au franchissement : {candidate.est_bbox_w:.0f}x{candidate.est_bbox_h:.0f}px

Question : ce véhicule a-t-il réellement traversé la ligne de comptage ?
Réponds UNIQUEMENT en JSON valide sans markdown :
{{"should_count": true/false, "reason": "explication courte", "direction": "↑ ou ↓", "cls": "car/truck/bicycle/etc"}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text   = response.content[0].text.strip()
        text   = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        result["from_cache"] = False

        cache[key] = {k: v for k, v in result.items() if k != "from_cache"}
        save_cache(target_date, cache)

        return result

    except Exception as e:
        log.error(f"Erreur Claude API : {e}")
        return {
            "should_count": False,
            "reason":       f"Erreur API : {e}",
            "direction":    candidate.direction,
            "cls":          candidate.cls,
            "from_cache":   False
        }

# ── Insertion en DB ────────────────────────────────────────────────────────────
def insert_missed(db_path: Path, candidate: Candidate,
                  cls: str, direction: str,
                  method: str = "rules",
                  reason: str = "") -> bool:
    if not candidate.cross_time:
        return False

    # Détermine le niveau de confiance
    n_frames = len([d for d in candidate.detections if d is not None])
    if method == "claude":
        confidence = "low"
    elif candidate.avg_conf >= 0.4 and n_frames >= 8:
        confidence = "high"
    elif candidate.avg_conf >= 0.25 and n_frames >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    try:
        conn = sqlite3.connect(str(db_path))

        # Vérifie si les colonnes confidence_level existent
        cols = [r[1] for r in conn.execute("PRAGMA table_info(passages)").fetchall()]

        if "confidence_level" in cols:
            conn.execute(
                """INSERT INTO passages
                   (timestamp, class, direction, track_id,
                    confidence_level, sanity_reason, n_frames, avg_conf)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    candidate.cross_time.strftime("%Y-%m-%d %H:%M:%S"),
                    cls,
                    direction,
                    -99,
                    confidence,
                    reason[:200] if reason else method,
                    n_frames,
                    round(candidate.avg_conf, 3),
                )
            )
        else:
            # Fallback si migration pas encore faite
            conn.execute(
                "INSERT INTO passages (timestamp, class, direction, track_id) "
                "VALUES (?,?,?,?)",
                (
                    candidate.cross_time.strftime("%Y-%m-%d %H:%M:%S"),
                    cls,
                    direction,
                    -99,
                )
            )

        conn.commit()
        conn.close()
        log.info(
            f"Inséré [{confidence}] : {cls} {direction} "
            f"@ {candidate.cross_time}  "
            f"frames={n_frames}  "
            f"conf={candidate.avg_conf:.2f}"
        )
        return True
    except Exception as e:
        log.error(f"Erreur insertion : {e}")
        return False

# ── Point d'entrée principal ───────────────────────────────────────────────────
def run_sanity_check(target_date: date = None,
                     min_detections: int = MIN_DETECTIONS,
                     duplicate_window: float = 5.0,
                     use_claude: bool = True) -> dict:

    if target_date is None:
        target_date = date.today()

    base_dir = Path(__file__).parent
    log_path = base_dir / "data" / f"counter_{target_date.isoformat()}.log"
    db_path  = base_dir / "data" / f"traffic_{target_date.isoformat()}.db"

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

    # Charge le cache Claude une seule fois
    claude_cache = load_cache(target_date) if use_claude else {}

    # Phase 4 — Analyse des manqués
    results_missed = []
    api_calls  = 0
    cache_hits = 0

    for cand in missed:
        if use_claude and cand.is_ambiguous:
            claude_result = analyze_with_claude(cand, claude_cache, target_date)
            if claude_result.get("from_cache"):
                cache_hits += 1
            else:
                api_calls += 1

            results_missed.append({
                "timestamp":    cand.cross_time.isoformat() if cand.cross_time else None,
                "cls":          claude_result["cls"],
                "direction":    claude_result["direction"],
                "avg_conf":     round(cand.avg_conf, 2),
                "n_frames":     len(cand.detections),
                "duration_sec": round(cand.duration_sec, 1),
                "should_count": claude_result["should_count"],
                "reason":       claude_result["reason"],
                "method":       "claude",
                "note":         cand.note,
                "from_cache":   claude_result.get("from_cache", False),
                "inserted":     False,
                "est_bbox_w":   round(cand.est_bbox_w, 0),
                "est_bbox_h":   round(cand.est_bbox_h, 0),
            })
        else:
            results_missed.append({
                "timestamp":    cand.cross_time.isoformat() if cand.cross_time else None,
                "cls":          cand.cls,
                "direction":    cand.direction,
                "avg_conf":     round(cand.avg_conf, 2),
                "n_frames":     len(cand.detections),
                "duration_sec": round(cand.duration_sec, 1),
                "should_count": not cand.is_ambiguous,
                "reason":       (
                    f"Ambigu — {len(cand.detections)} frames, conf={cand.avg_conf:.2f}"
                    if cand.is_ambiguous else
                    f"Séquence claire ({len(cand.detections)} frames, conf={cand.avg_conf:.2f})"
                ),
                "method":       "rules",
                "note":         cand.note,
                "inserted":     False,
                "est_bbox_w":   round(cand.est_bbox_w, 0),
                "est_bbox_h":   round(cand.est_bbox_h, 0),
            })

    # Phase 5 — Détection doublons
    duplicates = find_db_duplicates(db_passages, window_sec=duplicate_window)

    sort_bugs   = sum(1 for d in duplicates if d["type"] == "sort_bug")
    standard    = sum(1 for d in duplicates if d["type"] == "standard")
    manual_dups = sum(1 for d in duplicates if d["type"] == "manual_dup")

    slow_vehicles = [
        c for c in candidates if c.note and "lent" in c.note
    ]

    return {
        "date":              target_date.isoformat(),
        "log_path":          str(log_path),
        "db_path":           str(db_path),
        "total_yolo":        len(detections),
        "total_candidates":  len(candidates),
        "db_passages":       len(db_passages),
        "missed":            results_missed,
        "missed_count":      len([m for m in results_missed if m["should_count"]]),
        "duplicates":        duplicates,
        "duplicates_count":  len(duplicates),
        "sort_bugs":         sort_bugs,
        "standard_dups":     standard,
        "manual_dups":       manual_dups,
        "slow_vehicles":     [
            {
                "start":     c.start_time.strftime("%H:%M:%S") if c.start_time else "?",
                "end":       c.end_time.strftime("%H:%M:%S") if c.end_time else "?",
                "duration":  round(c.duration_sec, 1),
                "cls":       c.cls,
                "direction": c.direction,
                "note":      c.note,
            }
            for c in slow_vehicles
        ],
        "claude_api_calls":  api_calls,
        "claude_cache_hits": cache_hits,
    }

def insert_all_missed(report: dict) -> dict:
    db_path  = Path(report["db_path"])
    inserted = 0
    for item in report["missed"]:
        if item["should_count"] and not item["inserted"]:
            ts   = datetime.fromisoformat(item["timestamp"])
            cand = Candidate(
                cross_time=ts,
                cls=item["cls"],
                direction=item["direction"],
                avg_conf=item.get("avg_conf", 0.0),
                detections=[None] * item.get("n_frames", 0)
            )
            if insert_missed(
                db_path, cand,
                item["cls"], item["direction"],
                method=item.get("method", "rules"),
                reason=item.get("reason", "")
            ):
                item["inserted"] = True
                inserted += 1
    report["inserted_count"] = inserted
    return report

def remove_all_duplicates(report: dict) -> dict:
    db_path = Path(report["db_path"])
    removed = remove_db_duplicates(db_path, report["duplicates"])
    report["duplicates_removed"] = removed
    for dup in report["duplicates"]:
        dup["removed"] = True
    return report

# ── Cache des décisions Claude ─────────────────────────────────────────────────
def _candidate_key(candidate: Candidate) -> str:
    content = (
        f"{candidate.cls}"
        f"{candidate.direction}"
        f"{candidate.cross_time.isoformat() if candidate.cross_time else ''}"
        f"{candidate.avg_conf:.2f}"
        f"{len(candidate.detections)}"
        f"{candidate.detections[0].timestamp.isoformat() if candidate.detections else ''}"
    )
    return hashlib.md5(content.encode()).hexdigest()[:12]

def load_cache(target_date: date) -> dict:
    cache_path = (
        Path(__file__).parent / "data" /
        f"sanity_cache_{target_date.isoformat()}.json"
    )
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)
        log.info(f"Cache chargé : {len(cache)} décisions existantes")
        return cache
    return {}

def save_cache(target_date: date, cache: dict):
    cache_path = (
        Path(__file__).parent / "data" /
        f"sanity_cache_{target_date.isoformat()}.json"
    )
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    log.info(f"Cache sauvegardé : {len(cache)} décisions")

# ── Point d'entrée CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )

    parser = argparse.ArgumentParser(
        description="Sanity check — analyse les passages manqués et doublons"
    )
    parser.add_argument(
        "--date", default=date.today().isoformat(),
        help="Date à analyser (YYYY-MM-DD, défaut: aujourd'hui)"
    )
    parser.add_argument(
        "--min-detections", type=int, default=MIN_DETECTIONS,
        help=f"Minimum de frames YOLO pour un vrai passage (défaut: {MIN_DETECTIONS})"
    )
    parser.add_argument(
        "--duplicate-window", type=float, default=5.0,
        help="Fenêtre en secondes pour détecter les doublons (défaut: 5s)"
    )
    parser.add_argument(
        "--insert", action="store_true",
        help="Insérer les manqués + supprimer les doublons en DB"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyse uniquement, ne modifie pas la DB"
    )
    parser.add_argument(
        "--no-claude", action="store_true",
        help="Désactiver Claude API, utiliser uniquement les règles fixes"
    )
    args = parser.parse_args()

    target     = date.fromisoformat(args.date)
    use_claude = not args.no_claude

    print(f"\n{'='*60}")
    print(f"Sanity check — {target.isoformat()}")
    print(f"Min détections  : {args.min_detections}")
    print(f"Fenêtre doublons: {args.duplicate_window}s")
    print(f"Claude API      : {'activé' if use_claude else 'désactivé'}")
    print(f"Mode            : {'INSERT' if args.insert and not args.dry_run else 'DRY RUN'}")
    print(f"{'='*60}\n")

    report = run_sanity_check(
        target_date=target,
        min_detections=args.min_detections,
        duplicate_window=args.duplicate_window,
        use_claude=use_claude
    )

    if "error" in report:
        print(f"ERREUR : {report['error']}")
        sys.exit(1)

    if use_claude:
        print(
            f"Claude API calls  : {report.get('claude_api_calls', 0)}  "
            f"(cache hits: {report.get('claude_cache_hits', 0)})"
        )

    # ── Résumé général ──
    total_cands = report["total_candidates"]
    db_count    = report["db_passages"]
    rate        = round(db_count / total_cands * 100) if total_cands > 0 else 0

    print(f"Détections YOLO      : {report['total_yolo']}")
    print(f"Candidats passages   : {total_cands}")
    print(f"Passages en DB       : {db_count}")
    print(f"Taux de détection    : {rate}%")
    print(f"Passages manqués     : {report['missed_count']}")
    print(
        f"Doublons détectés    : {report['duplicates_count']} "
        f"({report['sort_bugs']} bugs SORT, "
        f"{report['standard_dups']} standard, "
        f"{report['manual_dups']} double-insertion)"
    )
    print(f"Véhicules lents      : {len(report['slow_vehicles'])}")

    # ── Passages manqués ──
    if report["missed"]:
        print(f"\n{'─'*60}")
        print(f"Passages manqués ({len(report['missed'])}) :\n")
        for i, m in enumerate(report["missed"], 1):
            ts   = m["timestamp"].split("T")[1][:8] if m["timestamp"] else "??:??:??"
            tag  = "COMPTER" if m["should_count"] else "IGNORER"
            slow = f"  [LENT {m['duration_sec']}s]" if m.get("duration_sec", 0) > SLOW_VEHICLE_SEC else ""
            print(
                f"  {i:2}. [{tag}]  {ts}  {m['cls']:<12}  {m['direction']}  "
                f"frames={m['n_frames']}  conf={m['avg_conf']}  "
                f"({m['method']}){slow}"
            )
            print(f"       {m['reason']}")
            if m.get("note"):
                print(f"       NOTE: {m['note']}")
    else:
        print("\nAucun passage manqué.")

    # ── Doublons ──
    if report["duplicates"]:
        print(f"\n{'─'*60}")
        print(f"Doublons ({report['duplicates_count']}) :\n")
        for i, dup in enumerate(report["duplicates"], 1):
            type_map = {
                "sort_bug":   "BUG SORT",
                "standard":   "DOUBLON",
                "manual_dup": "DOUBLE INSERTION",
            }
            type_label = type_map.get(dup["type"], dup["type"].upper())
            print(
                f"  {i:2}. [{type_label}]  "
                f"GARDER id={dup['keep']['id']:<6} "
                f"{dup['keep']['class']:<12} {dup['keep']['direction']}  "
                f"@ {dup['keep']['timestamp']}"
            )
            print(
                f"            SUPPR. id={dup['remove']['id']:<6} "
                f"{dup['remove']['class']:<12} {dup['remove']['direction']}  "
                f"@ {dup['remove']['timestamp']}  "
                f"({dup['reason']})"
            )
    else:
        print("\nAucun doublon détecté.")

    # ── Véhicules lents ──
    if report["slow_vehicles"]:
        print(f"\n{'─'*60}")
        print(f"Véhicules lents ({len(report['slow_vehicles'])}) :\n")
        for sv in report["slow_vehicles"]:
            print(
                f"  {sv['start']} → {sv['end']}  "
                f"{sv['cls']:<12}  {sv['direction']}  "
                f"{sv['duration']}s"
            )
            print(f"  {sv['note']}")

    # ── Actions ──
    if args.insert and not args.dry_run:
        print(f"\n{'─'*60}")

        to_insert = [m for m in report["missed"] if m["should_count"]]
        if to_insert:
            print(f"Insertion de {len(to_insert)} passage(s) manqué(s)...")
            report = insert_all_missed(report)
            print(f"  → {report['inserted_count']} passage(s) inséré(s)")
        else:
            print("Aucun passage à insérer.")

        if report["duplicates"]:
            print(f"Suppression de {report['duplicates_count']} doublon(s)...")
            report = remove_all_duplicates(report)
            print(f"  → {report['duplicates_removed']} doublon(s) supprimé(s)")
        else:
            print("Aucun doublon à supprimer.")
    else:
        actions = []
        to_insert = [m for m in report["missed"] if m["should_count"]]
        if to_insert:
            actions.append(f"{len(to_insert)} passage(s) seraient insérés")
        if report["duplicates"]:
            actions.append(f"{report['duplicates_count']} doublon(s) seraient supprimés")
        if actions:
            print(f"\n{'─'*60}")
            print(f"DRY RUN — {' + '.join(actions)}")
            print("Relance avec --insert pour appliquer les changements.")

    print(f"\n{'='*60}\n")