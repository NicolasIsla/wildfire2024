import gdown
import zipfile
import os
import shutil

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

            # Extraer el contenido del ZIP a una carpeta temporal
            temp_extract_path = os.path.join(destination, 'temp')
            os.makedirs(temp_extract_path, exist_ok=True)
            
            print(f"Extrayendo {info['name']}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_path)

            # Mover archivos desde la carpeta temporal al destino final
            for root, _, files in os.walk(temp_extract_path):
                for file in files:
                    src = os.path.join(root, file)
                    dst = os.path.join(destination, file)
                    shutil.move(src, dst)

            # Limpiar la carpeta temporal y eliminar el ZIP
            shutil.rmtree(temp_extract_path)
            os.remove(output_path)
            print(f"{info['name']} extraído y ZIP eliminado.")

        print("Todas las partes descargadas, extraídas y procesadas exitosamente.")
    except Exception as e:
        print(f"Error durante el proceso: {e}")

if __name__ == "__main__":
    # Define las URLs y nombres de los archivos a descargar
    file_urls = {
        # URLs de ejemplo
        'part1': {'id': '1BzrKREedNtPsZukliB1Sl64e9e6txinQ', 'name': 'train_part1'},
        
    }

    # Ruta de destino donde se guardarán los archivos extraídos
    destination = "/data/nisla/Smoke50v3/DS/labels"

    # Ejecutar la función para descargar, extraer y eliminar progresivamente
    download_and_extract(file_urls, destination)

