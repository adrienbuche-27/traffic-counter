#!/bin/bash
# run_counter.sh — lance live_counter.py par cycles de 59min avec 1min de pause

LOG="/home/abuche/Documents/traffic-counter/data/runner.log"
VENV="/home/abuche/Documents/traffic-counter/traffic/bin/python"
SCRIPT="/home/abuche/Documents/traffic-counter/live_counter.py"
WORKDIR="/home/abuche/Documents/traffic-counter"

RUN_MINUTES=10
COOLDOWN_SECONDS=10

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S')  $1" | tee -a "$LOG"
}

log "=============================="
log "Démarrage run_counter.sh"
log "Cycle : ${RUN_MINUTES}min actif / ${COOLDOWN_SECONDS}s pause"
log "=============================="

while true; do

    log "Lancement de live_counter.py..."
    cd "$WORKDIR"

    # Lance le script avec timeout — SIGTERM propre après 59min
    timeout "${RUN_MINUTES}m" "$VENV" "$SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 124 ]; then
        log "Timeout atteint (${RUN_MINUTES}min) — arrêt normal"
    elif [ $EXIT_CODE -eq 0 ]; then
        log "live_counter.py terminé proprement (code 0)"
    else
        log "live_counter.py terminé avec code $EXIT_CODE"
    fi

    log "Pause de ${COOLDOWN_SECONDS}s pour refroidissement..."
    sleep "$COOLDOWN_SECONDS"

    log "Redémarrage du cycle..."

done
