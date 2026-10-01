#!/usr/bin/env bash
set -euo pipefail

EXECUTABLE="./overlap_vec_mat_reduce"
DATA_BASE="../data"
TEST_BASE="../testing"

CONFIGS=("1_1" "4_4" "8_8")
SIZES=("small" "medium" "large")
VEC_FACTORS=(1 2 4 8 16)

WARMUP_RUNS=2
MEASURED_RUNS=4

WARM_UP=1
NOT_WARM_UP=0

for cfg in "${CONFIGS[@]}"; do
    for sz in "${SIZES[@]}"; do
        DATASET_H="${DATA_BASE}/${cfg}_data/${sz}_data/habitat/"
        DATASET_F="${DATA_BASE}/${cfg}_data/${sz}_data/flow/"
        
        # Crea la cartella di output se non esiste
        mkdir -p "${TEST_BASE}/${cfg}_test"
        TEST_JSON="${TEST_BASE}/${cfg}_test/${sz}_test.json"

        # Verifica preliminare dell'esistenza delle cartelle
        if [[ ! -d "$DATASET_H" ]] || [[ ! -d "$DATASET_F" ]]; then
            echo "[SKIP] Percorsi non trovati: $DATASET_H o $DATASET_F"
            continue
        fi

        echo "=========================================================="
        echo " Configurazione: ${cfg} | Taglia: ${sz}"
        echo " Habitat:   ${DATASET_H}"
        echo " Flow:      ${DATASET_F}"
        echo " Test JSON: ${TEST_JSON}"
        echo "=========================================================="

        for vec in "${VEC_FACTORS[@]}"; do
            echo "--- Test con vettorizzazione factor: ${vec} ---"

            # Warm-up (con $TEST_JSON come quinto argomento)
            for ((w = 1; w <= WARMUP_RUNS; w++)); do
                echo "  [Warmup $w/$WARMUP_RUNS]"
                "$EXECUTABLE" "$DATASET_H" "$DATASET_F" "$vec" "$WARM_UP" "$TEST_JSON" > /dev/null 2>&1
            done

            # Esecuzioni misurate (con $TEST_JSON come quinto argomento)
            for ((i = 0; i < MEASURED_RUNS; i++)); do
                echo "  [Run $i]"
                "$EXECUTABLE" "$DATASET_H" "$DATASET_F" "$vec" "$NOT_WARM_UP" "$TEST_JSON"
            done
        done
        echo ""
    done
done

echo "Tutti i benchmark sono stati completati con successo."