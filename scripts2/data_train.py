import gdown
import zipfile
import os
import argparse

import gdown
import zipfile
import os

def download_and_extract(file_urls, destination):
    """
    Descarga cada parte de un archivo desde Google Drive, la extrae y elimina el ZIP inmediatamente.
    """
    try:
        # Crear el directorio de destino si no existe
        if not os.path.exists(destination):
            os.makedirs(destination)

        # Procesar cada archivo en la lista de URLs
        for part_name, info in file_urls.items():
            output_path = os.path.join(destination, f"{info['name']}.zip")
            print(f"Descargando {info['name']}...")

            # Descargar el archivo ZIP
            gdown.download(f"https://drive.google.com/uc?id={info['id']}", output_path, quiet=False)

            # Extraer el contenido del ZIP
            print(f"Extrayendo {info['name']}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(destination)

            # Eliminar el archivo ZIP una vez extraído
            os.remove(output_path)
            print(f"{info['name']} extraído y ZIP eliminado.")

        print("Todas las partes descargadas, extraídas y procesadas exitosamente.")
    except Exception as e:
        print(f"Error durante el proceso: {e}")

if __name__ == "__main__":
    # Define las URLs y nombres de los archivos a descargar
    file_urls = {
        # URLs de ejemplo
        'part1': {'id': '1fgFh81Uu5rp8GnZzk7GFLmWdvsIeDBk0', 'name': 'train_part1'},
        'part2': {'id': '1gkr6yFR1Dj_Mht_gUAH8vW_RWzt2om8g', 'name': 'train_part2'}
        # Agrega más partes aquí si es necesario
    }

    # Ruta de destino donde se guardarán los archivos extraídos
    destination = "/data/nisla/Smoke50v3/DS/images/train"

    # Ejecutar la función para descargar, extraer y eliminar progresivamente
    download_and_extract(file_urls, destination)
