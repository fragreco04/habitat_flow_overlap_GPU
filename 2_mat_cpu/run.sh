#!/usr/bin/env bash
set -euo pipefail

# Parametri generali ed eseguibile
EXECUTABLE="./overlap_cpu_mat"
DATA_BASE_DIR="../data"
TESTING_BASE_DIR="../testing"

# Configurazioni di test
GRID_TYPES=("1_1" "4_4" "8_8")
SIZES=("small" "medium" "large")

VEC_FACTORS=(1)
WARMUP_RUNS=2
MEASURED_RUNS=4

WARM_UP=1
NOT_WARM_UP=0

for grid in "${GRID_TYPES[@]}"; do
    data_grid_dir="${DATA_BASE_DIR}/${grid}_data"
    test_grid_dir="${TESTING_BASE_DIR}/${grid}_test"

    # Assicura che la directory di destinazione dei test esista
    mkdir -p "$test_grid_dir"

    for size in "${SIZES[@]}"; do
        dataset_h="${data_grid_dir}/${size}_data/habitat/"
        dataset_f="${data_grid_dir}/${size}_data/flow/"
        testing_output="${test_grid_dir}/${size}_test.json"

        echo "=========================================================="
        echo " Avvio test: Grid ${grid} | Taglia: ${size}"
        echo " Dataset H: $dataset_h"
        echo " Dataset F: $dataset_f"
        echo " Output:    $testing_output"
        echo "=========================================================="

        for vec in "${VEC_FACTORS[@]}"; do
            echo "--- Vettorizzazione factor: $vec ---"

            # Giri di warm-up (2 run per configurazione)
            for ((w = 1; w <= WARMUP_RUNS; w++)); do
                echo "  [Warmup $w/$WARMUP_RUNS]"
                "$EXECUTABLE" "$dataset_h" "$dataset_f" "$WARM_UP" "$testing_output"
            done

            # Esecuzioni effettive misurate
            for ((i = 0; i < MEASURED_RUNS; i++)); do
                echo "  [Run $i/$MEASURED_RUNS]"
                "$EXECUTABLE" "$dataset_h" "$dataset_f" "$NOT_WARM_UP" "$testing_output"
            done
        done
        echo ""
    done
done

echo "Tutti i benchmark sono stati completati con successo."