import argparse
import os
import shutil
import numpy as np


def generate_sparse_matrix(rows, cols, min_density=0.20, max_density=0.40):
    """
    Genera una matrice float sparsa con valori > 0 solo dove la maschera è attiva.
    Garantisce una densità compresa nell'intervallo [min_density, max_density]
    e assicura che ci sia sempre almeno una cella attiva (> 0.0).
    """
    total_elements = rows * cols
    
    # Sceglie una densità casuale all'interno del range per questa specifica matrice
    current_density = np.random.uniform(min_density, max_density)
    
    # Calcola il numero esatto di celle attive garantendo che sia almeno 1
    num_active = int(round(current_density * total_elements))
    num_active = max(1, min(num_active, total_elements))

    # Seleziona casualmente gli indici delle celle da attivare
    flat_indices = np.random.choice(total_elements, size=num_active, replace=False)
    
    # Inizializza la matrice a zero e assegna valori positivi solo agli indici scelti
    matrix = np.zeros(total_elements, dtype=float)
    matrix[flat_indices] = np.random.uniform(0.1, 10.0, size=num_active)
    
    return matrix.reshape((rows, cols))


def save_matrix(filepath, matrix):
    """Salva la matrice in formato testo: colonne separate da spazio, senza metadati."""
    np.savetxt(filepath, matrix, fmt="%.4f", delimiter=" ")


def clean_directory(dir_path):
    """Svuota il contenuto di una cartella senza cancellare la cartella stessa."""
    os.makedirs(dir_path, exist_ok=True)
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generatore di dataset di test per overlap tra Habitat e Flow."
    )
    parser.add_argument(
        "-n",
        "--num_habitat",
        type=int,
        required=True,
        help="Numero di file Habitat (n)",
    )
    parser.add_argument(
        "-m", "--num_flow", type=int, required=True, help="Numero di file Flow (m)"
    )
    parser.add_argument(
        "-r", "--rows", type=int, required=True, help="Numero di righe delle matrici"
    )
    parser.add_argument(
        "-c",
        "--cols",
        type=int,
        required=True,
        help="Numero di colonne delle matrici",
    )
    parser.add_argument(
        "--min_density",
        type=float,
        default=0.20,
        help="Densità minima di celle attive (default: 0.20)",
    )
    parser.add_argument(
        "--max_density",
        type=float,
        default=0.40,
        help="Densità massima di celle attive (default: 0.40)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed casuale opzionale per riproducibilità",
    )

    args = parser.parse_args()

    if args.min_density > args.max_density:
        parser.error("--min_density non può essere maggiore di --max_density")

    if args.seed is not None:
        np.random.seed(args.seed)

    # Percorsi file e directory
    dir_habitat = os.path.join("data", "habitat")
    dir_flow = os.path.join("data", "flow")
    gt_path = os.path.join("data", "ground_truth.txt")

    # Pulizia delle directory di output
    clean_directory(dir_habitat)
    clean_directory(dir_flow)

    # Rimozione del file ground_truth precedente se presente
    if os.path.exists(gt_path):
        os.remove(gt_path)

    habitat_matrices = []
    flow_matrices = []

    # Generazione e salvataggio file Habitat
    for i in range(args.num_habitat):
        mat = generate_sparse_matrix(
            args.rows, args.cols, args.min_density, args.max_density
        )
        habitat_matrices.append(mat)
        path = os.path.join(dir_habitat, f"habitat_{i}.txt")
        save_matrix(path, mat)

    # Generazione e salvataggio file Flow
    for j in range(args.num_flow):
        mat = generate_sparse_matrix(
            args.rows, args.cols, args.min_density, args.max_density
        )
        flow_matrices.append(mat)
        path = os.path.join(dir_flow, f"flow_{j}.txt")
        save_matrix(path, mat)

    # Calcolo della matrice di Ground Truth (Overlap Ratio)
    # Dimensioni: n (Habitat, righe) x m (Flow, colonne)
    ground_truth = np.zeros((args.num_habitat, args.num_flow), dtype=float)

    for i in range(args.num_habitat):
        h_active = habitat_matrices[i] > 0.0
        h_count = np.count_nonzero(h_active)

        for j in range(args.num_flow):
            f_active = flow_matrices[j] > 0.0
            intersection_count = np.count_nonzero(h_active & f_active)
            ground_truth[i, j] = intersection_count / h_count

    # Salvataggio ground truth
    save_matrix(gt_path, ground_truth)

    print(f"Generazione completata con successo:")
    print(f" - {args.num_habitat} file Habitat in '{dir_habitat}/'")
    print(f" - {args.num_flow} file Flow in '{dir_flow}/'")
    print(f" - Range di densità attivo: [{args.min_density:.2%}, {args.max_density:.2%}]")
    print(
        f" - Matrice Ground Truth ({args.num_habitat}x{args.num_flow}) salvata in '{gt_path}'"
    )


if __name__ == "__main__":
    main()