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
        'part1': {'id': '1fgFh81Uu5rp8GnZzk7GFLmWdvsIeDBk0', 'name': 'train_part1'},
        'part2': {'id': '1gkr6yFR1Dj_Mht_gUAH8vW_RWzt2om8g', 'name': 'train_part2'},
        # https://drive.google.com/file/d/1mOZNk7pQmBK3vNoU-VXL9kmDo3cKBNOa/view?usp=drive_link
        'part3': {'id': '1mOZNk7pQmBK3vNoU-VXL9kmDo3cKBNOa', 'name': 'train_part3'},
        # https://drive.google.com/file/d/1rN86Xi9JOSZjVe9c4dMbZzFmZn6OK4Xp/view?usp=drive_link
        'part4': {'id': '1rN86Xi9JOSZjVe9c4dMbZzFmZn6OK4Xp', 'name': 'train_part4'},
        # https://drive.google.com/file/d/13JnWRFU0HejJ9agQswOv-v_4JDeL3eQF/view?usp=drive_link
        'part5': {'id': '13JnWRFU0HejJ9agQswOv-v_4JDeL3eQF', 'name': 'train_part5'},
        # https://drive.google.com/file/d/1AVignIUhd_Lf9RqOrlfavTTxFoVs_nuT/view?usp=drive_link
        'part6': {'id': '1AVignIUhd_Lf9RqOrlfavTTxFoVs_nuT', 'name': 'train_part6'},
        # https://drive.google.com/file/d/1vmlbysyoHZ8sVe-lhOE5k45b-REd3Qnm/view?usp=drive_link
        'part7': {'id': '1vmlbysyoHZ8sVe-lhOE5k45b-REd3Qnm', 'name': 'train_part7'},
        # https://drive.google.com/file/d/1rZuV9WBHgEmo6Eq2_oO8qLrbjDcmEG9p/view?usp=drive_link
        'part8': {'id': '1rZuV9WBHgEmo6Eq2_oO8qLrbjDcmEG9p', 'name': 'train_part8'},
        # https://drive.google.com/file/d/1XAgl5v96enRjMCQ8BNzkOlok0HMsxL2R/view?usp=drive_link
        'part9': {'id': '1XAgl5v96enRjMCQ8BNzkOlok0HMsxL2R', 'name': 'train_part9'},
        # https://drive.google.com/file/d/1b9_jJybQOYE6Gcapb5SICVhycpiMSL-0/view?usp=drive_link
        'part10': {'id': '1b9_jJybQOYE6Gcapb5SICVhycpiMSL-0', 'name': 'train_part10'},
        
    }

    # Ruta de destino donde se guardarán los archivos extraídos
    destination = "/data/nisla/Smoke50v3/DS/images/train"

    # Ejecutar la función para descargar, extraer y eliminar progresivamente
    download_and_extract(file_urls, destination)

