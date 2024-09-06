import os
import argparse
from PIL import Image
import imagehash
import pandas as pd

def calcular_hash(directorio, tipo):
    hashes = []
    paths = []
    tipos = []
    
    # Recorrer el directorio y calcular el hash de cada imagen
    for subdir, dirs, files in os.walk(directorio):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):  # Ajusta los formatos según sea necesario
                path = os.path.join(subdir, file)
                image = Image.open(path)
                hash = imagehash.average_hash(image)  # Cambia el tipo de hash si prefieres
                hashes.append(str(hash))  # Convertir a string para que sea hashable
                paths.append(path)
                tipos.append(tipo)

    return pd.DataFrame({'path': paths, 'hash': hashes, 'type': tipos})

def main(base_path):
    # Definir los subdirectorios de train, val, y test
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    test_dir = os.path.join(base_path, 'test')

    # Calcular hashes para cada conjunto
    train_hashes = calcular_hash(train_dir, 'train')
    val_hashes = calcular_hash(val_dir, 'val')
    test_hashes = calcular_hash(test_dir, 'test')

    # Combinar los DataFrames
    all_hashes = pd.concat([train_hashes, val_hashes, test_hashes])

    # Agrupar las imágenes por hash y listar los paths y tipos
    grouped = all_hashes.groupby('hash').agg(list)

    print(grouped)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calcular y agrupar hashes de imágenes entre conjuntos de datos.')
    parser.add_argument('base_path', type=str, help='El path base que contiene los directorios train, val y test')
    
    args = parser.parse_args()
    
    main(args.base_path)