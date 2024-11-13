import json
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

# Función para calcular distancia de Hamming entre dos hashes
def hamming_distance(hash1, hash2):
    bin1 = bin(int(hash1, 16))[2:].zfill(256)
    bin2 = bin(int(hash2, 16))[2:].zfill(256)
    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

# Función para procesar pares dentro de un dataset
def process_same_dataset(args):
    subset_name, images = args
    results = []
    paths = list(images.keys())
    hashes = list(images.values())
    for (hash1, hash2), (path1, path2) in zip(itertools.combinations(hashes, 2), itertools.combinations(paths, 2)):
        distance = hamming_distance(hash1, hash2)
        results.append({
            'subset': subset_name,
            'path1': path1,
            'path2': path2,
            'distance': distance
        })
    return results

# Función para procesar pares entre datasets
def process_cross_dataset(args):
    subset_name1, images1, subset_name2, images2 = args
    results = []
    paths1 = list(images1.keys())
    hashes1 = list(images1.values())
    paths2 = list(images2.keys())
    hashes2 = list(images2.values())
    for hash1, path1 in zip(hashes1, paths1):
        for hash2, path2 in zip(hashes2, paths2):
            distance = hamming_distance(hash1, hash2)
            results.append({
                'subset1': subset_name1,
                'path1': path1,
                'subset2': subset_name2,
                'path2': path2,
                'distance': distance
            })
    return results

if __name__ == '__main__':
    # Cargar múltiples archivos JSON
    datasets = {}
    json_files = [
        '/data/nisla/SmokesFrames-2.4k/DS/images_hashes.json', 
        "/data/nisla/Nemo/DS/images_hashes.json"
    ]
    for file in json_files:
        with open(file, 'r') as f:
            datasets[file] = json.load(f)

    # Preparar datos para procesamiento paralelo (distancias dentro del mismo dataset)
    same_dataset_tasks = []
    for dataset_name, dataset in datasets.items():
        for subset_name, images in dataset.items():
            same_dataset_tasks.append((subset_name, images))

    # Procesar distancias dentro de cada dataset con multiprocessing
    with Pool(cpu_count()) as pool:
        same_dataset_results = pool.map(process_same_dataset, same_dataset_tasks)

    # Aplanar resultados de las distancias dentro del mismo dataset
    same_dataset_results = [item for sublist in same_dataset_results for item in sublist]

    # Preparar datos para procesamiento paralelo (distancias entre datasets)
    cross_dataset_tasks = []
    for (dataset_name1, dataset1), (dataset_name2, dataset2) in itertools.combinations(datasets.items(), 2):
        for subset_name1, images1 in dataset1.items():
            for subset_name2, images2 in dataset2.items():
                cross_dataset_tasks.append((subset_name1, images1, subset_name2, images2))

    # Procesar distancias entre datasets con multiprocessing
    with Pool(cpu_count()) as pool:
        cross_dataset_results = pool.map(process_cross_dataset, cross_dataset_tasks)

    # Aplanar resultados de las distancias entre datasets
    cross_dataset_results = [item for sublist in cross_dataset_results for item in sublist]

    # Crear DataFrames con los resultados
    df_same_dataset = pd.DataFrame(same_dataset_results)
    df_cross_dataset = pd.DataFrame(cross_dataset_results)

    # Guardar resultados a CSV
    df_same_dataset.to_csv('hamming_distances_same_dataset.csv', index=False)
    df_cross_dataset.to_csv('hamming_distances_cross_datasets.csv', index=False)

    print("CSV generados con éxito:")
    print("1. hamming_distances_same_dataset.csv")
    print("2. hamming_distances_cross_datasets.csv")

