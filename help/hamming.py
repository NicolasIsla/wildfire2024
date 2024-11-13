# import json
# import itertools
# import pandas as pd

# # Función para calcular distancia de Hamming entre dos hashes
# def hamming_distance(hash1, hash2):
#     # Convertir los hashes de hexadecimal a binario
#     bin1 = bin(int(hash1, 16))[2:].zfill(256)
#     bin2 = bin(int(hash2, 16))[2:].zfill(256)
#     # Contar diferencias bit a bit
#     return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))

# # Cargar múltiples archivos JSON
# datasets = {}
# json_files = ['images_hashes.json', 'images_hashes Copy.json']  # Actualiza los nombres y ubicaciones de los archivos
# for file in json_files:
#     with open(file, 'r') as f:
#         datasets[file] = json.load(f)

# # Calcular distancias dentro de cada dataset
# same_dataset_results = []
# for dataset_name, dataset in datasets.items():
#     for subset_name, images in dataset.items():
#         hashes = list(images.values())
#         for hash1, hash2 in itertools.combinations(hashes, 2):
#             distance = hamming_distance(hash1, hash2)
#             same_dataset_results.append({
#                 'dataset': dataset_name,
#                 'subset': subset_name,
#                 'hash1': hash1,
#                 'hash2': hash2,
#                 'distance': distance
#             })

# # Calcular distancias entre datasets
# cross_dataset_results = []
# for (dataset_name1, dataset1), (dataset_name2, dataset2) in itertools.combinations(datasets.items(), 2):
#     for subset_name1, images1 in dataset1.items():
#         for subset_name2, images2 in dataset2.items():
#             for hash1 in images1.values():
#                 for hash2 in images2.values():
#                     distance = hamming_distance(hash1, hash2)
#                     cross_dataset_results.append({
#                         'dataset1': dataset_name1,
#                         'subset1': subset_name1,
#                         'hash1': hash1,
#                         'dataset2': dataset_name2,
#                         'subset2': subset_name2,
#                         'hash2': hash2,
#                         'distance': distance
#                     })

# # Crear DataFrames con los resultados
# df_same_dataset = pd.DataFrame(same_dataset_results)
# df_cross_dataset = pd.DataFrame(cross_dataset_results)

# # Guardar resultados a CSV
# df_same_dataset.to_csv('hamming_distances_same_dataset.csv', index=False)
# df_cross_dataset.to_csv('hamming_distances_cross_datasets.csv', index=False)

# print("CSV generados con éxito:")
# print("1. hamming_distances_same_dataset.csv")
# print("2. hamming_distances_cross_datasets.csv")
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
    subset_name, hashes = args
    results = []
    for hash1, hash2 in itertools.combinations(hashes, 2):
        distance = hamming_distance(hash1, hash2)
        results.append({
            'subset': subset_name,
            'hash1': hash1,
            'hash2': hash2,
            'distance': distance
        })
    return results

# Función para procesar pares entre datasets
def process_cross_dataset(args):
    subset_name1, hashes1, subset_name2, hashes2 = args
    results = []
    for hash1 in hashes1:
        for hash2 in hashes2:
            distance = hamming_distance(hash1, hash2)
            results.append({
                'subset1': subset_name1,
                'hash1': hash1,
                'subset2': subset_name2,
                'hash2': hash2,
                'distance': distance
            })
    return results

if __name__ == '__main__':
    # Cargar múltiples archivos JSON
    datasets = {}
    json_files = ['images_hashes.json', 'images_hashes Copy.json']
    for file in json_files:
        with open(file, 'r') as f:
            datasets[file] = json.load(f)

    # Preparar datos para procesamiento paralelo (distancias dentro del mismo dataset)
    same_dataset_tasks = []
    for dataset_name, dataset in datasets.items():
        for subset_name, images in dataset.items():
            hashes = list(images.values())
            same_dataset_tasks.append((subset_name, hashes))

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
                hashes1 = list(images1.values())
                hashes2 = list(images2.values())
                cross_dataset_tasks.append((subset_name1, hashes1, subset_name2, hashes2))

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
