import os
import json
from PIL import Image
import imagehash

def calculate_phash(image_path, hash_size=16):
    """
    Calcula el perceptual hash (pHash) de una imagen con un tamaño de hash ajustable.
    """
    try:
        with Image.open(image_path) as img:
            phash = imagehash.phash(img, hash_size=hash_size)
        return str(phash)  # Convertir el hash a string para almacenamiento
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def process_datasets(base_path, output_file='image_hashes.json'):
    """
    Procesa los datasets y guarda los hashes de cada imagen en un archivo JSON.
    """
    datasets = ['train', 'val', 'test']
    hash_data = {}
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.exists(dataset_path):
            print(f"Ruta no encontrada: {dataset_path}")
            continue
        
        print(f"Procesando dataset: {dataset}")
        hash_data[dataset] = {}
        for image_file in os.listdir(dataset_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                print(f"Procesando imagen: {image_file}")
                image_path = os.path.join(dataset_path, image_file)
                image_hash = calculate_phash(image_path)
                if image_hash:  # Solo agregar si no hubo error
                    hash_data[dataset][image_file] = image_hash

    # Guardar hashes en un archivo JSON
    with open(output_file, 'w') as json_file:
        json.dump(hash_data, json_file, indent=4)
    print(f"Hashes guardados en {output_file}")

# Rutas base de los datasets
dataset_paths = [
    r"/data/nisla/Nemo/DS/images",
    r"/data/nisla/SmokesFrames-2.4k/DS/images",
    r"/data/nisla/Smoke50v3/DS/images/"
]

# Ejecutar el script
for dataset_path in dataset_paths:
    # Nombre del archivo de salida a partir de la ruta del dataset
    output_file = (dataset_path).replace(" ", "_") + '_hashes.json'
    process_datasets(dataset_path, output_file)

