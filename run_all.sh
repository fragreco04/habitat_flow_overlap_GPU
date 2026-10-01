#!/usr/bin/env bash
set -euo pipefail

# Lista delle cartelle da eseguire in sequenza
DIRECTORIES=(
    "2_mat_cpu"
    "3_vec_mat"
    "4_vec_mat_reduce"
    "5_vec_mat_reduce_ott"
)

SCRIPT_NAME="run.sh"

echo "=== Reset dei file JSON dal template ==="
for grid in "1_1" "4_4" "8_8"; do
    mkdir -p "testing/${grid}_test"
    for size in "small" "medium" "large"; do
        cp testing/template.json "testing/${grid}_test/${size}_test.json"
    done
done
echo ""

echo "=== Inizio esecuzione sequenziale dei benchmark ==="
echo ""

for dir in "${DIRECTORIES[@]}"; do
    TARGET_SCRIPT="${dir}/${SCRIPT_NAME}"

    if [[ ! -d "$dir" ]]; then
        echo "[-] ERRORE: La cartella '$dir' non esiste. Salto."
        continue
    fi

    if [[ ! -f "$TARGET_SCRIPT" ]]; then
        echo "[-] ATTENZIONE: File '$TARGET_SCRIPT' non trovato. Salto."
        continue
    fi

    echo "=========================================================="
    echo ">> Avvio benchmark in: $dir"
    echo "=========================================================="

    (
        cd "$dir"
        bash "$SCRIPT_NAME"
    )

    echo ""
    echo ">> Completato: $dir"
    echo ""
done

echo "=== Tutti i benchmark sono stati completati con successo! ==="