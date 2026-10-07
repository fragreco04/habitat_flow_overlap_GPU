#!/usr/bin/env bash
set -euo pipefail

TESTING_DIR="../testing"
PROGRAMS_DIR="programs"

# Lista delle sottocartelle dentro programs/ da eseguire in sequenza
DIRECTORIES=(
    "2_mat_cpu"
    "3_vec_mat"
    "4_vec_mat_reduce"
    "5_vec_mat_reduce_ott"
)

SCRIPT_NAME="run.sh"

echo "=== Reset dei file JSON dal template ==="
for grid in "1_1" "4_4" "8_8"; do
    mkdir -p "${TESTING_DIR}/${grid}_test"
    for size in "small" "medium" "large"; do
        cp "${TESTING_DIR}/template.json" "${TESTING_DIR}/${grid}_test/${size}_test.json"
    done
done
echo ""

echo "=== Inizio esecuzione sequenziale dei benchmark ==="
echo ""

for dir in "${DIRECTORIES[@]}"; do
    TARGET_DIR="${PROGRAMS_DIR}/${dir}"
    TARGET_SCRIPT="${TARGET_DIR}/${SCRIPT_NAME}"

    if [[ ! -d "$TARGET_DIR" ]]; then
        echo "[-] ERRORE: La cartella '$TARGET_DIR' non esiste. Salto."
        continue
    fi

    if [[ ! -f "$TARGET_SCRIPT" ]]; then
        echo "[-] ATTENZIONE: File '$TARGET_SCRIPT' non trovato. Salto."
        continue
    fi

    echo "=========================================================="
    echo ">> Avvio benchmark in: $TARGET_DIR"
    echo "=========================================================="

    (
        cd "$TARGET_DIR"
        bash "$SCRIPT_NAME"
    )

    echo ""
    echo ">> Completato: $TARGET_DIR"
    echo ""
done

echo "=== Tutti i benchmark sono stati completati con successo! ==="