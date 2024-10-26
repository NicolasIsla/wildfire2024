import gdown
import zipfile
import os
import argparse

def download_and_merge(file_urls, destination):
    """
    Descarga múltiples partes de un archivo desde Google Drive y las reconstruye.
    """
    try:
        # Crear el directorio de destino si no existe
        if not os.path.exists(destination):
            os.makedirs(destination)

        # Descargar cada parte del archivo
        part_files = []
        for part_name, info in file_urls.items():
            output_path = os.path.join(destination, f"{info['name']}.zip")
            print(f"Descargando {info['name']}...")
            gdown.download(f"https://drive.google.com/uc?id={info['id']}", output_path, quiet=False)
            part_files.append(output_path)

        # Extraer y combinar las partes
        combined_zip_path = os.path.join(destination, 'combined.zip')
        with zipfile.ZipFile(combined_zip_path, 'w') as combined_zip:
            for part_file in part_files:
                with zipfile.ZipFile(part_file, 'r') as part_zip:
                    for file_name in part_zip.namelist():
                        combined_zip.writestr(file_name, part_zip.read(file_name))
                os.remove(part_file)  # Eliminar la parte descargada

        # Descomprimir el archivo combinado si es necesario
        with zipfile.ZipFile(combined_zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # Eliminar el archivo combinado ZIP
        os.remove(combined_zip_path)

        print("Archivo reconstruido y extraído exitosamente.")
    except Exception as e:
        print(f"Error al descargar o reconstruir: {e}")

if __name__ == "__main__":
    # Define las URLs y nombres de los archivos a descargar
    file_urls = {
        # https://drive.google.com/file/d/1fgFh81Uu5rp8GnZzk7GFLmWdvsIeDBk0/view?usp=drive_link
        'part1': {'id': '1fgFh81Uu5rp8GnZzk7GFLmWdvsIeDBk0', 'name': 'train_part1'},
        # https://drive.google.com/file/d/1gkr6yFR1Dj_Mht_gUAH8vW_RWzt2om8g/view?usp=drive_link
        'part2': {'id': '1gkr6yFR1Dj_Mht_gUAH8vW_RWzt2om8g', 'name': 'train_part2'},
        # 'part3': {'id': 'ID_DE_GOOGLE_DRIVE_3', 'name': 'train_part3'},
        # 'part4': {'id': 'ID_DE_GOOGLE_DRIVE_4', 'name': 'train_part4'},
        # 'part5': {'id': 'ID_DE_GOOGLE_DRIVE_5', 'name': 'train_part5'}
    }

    # Define la ruta de destino donde se guardarán los archivos
    destination = "/data/nisla/Smoke50v3/DS/images/train"

    # Llamar a la función para descargar y reconstruir
    download_and_merge(file_urls, destination)
