import gdown
import zipfile
import os

# Función para descargar el archivo
def download_file(url, output_path):
    gdown.download(url, output_path, fuzzy=True)
    print(f"Archivo descargado: {output_path}")

# Función para descomprimir el archivo
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Archivos descomprimidos en: {extract_to}")

# URL del archivo en Google Drive
file_url = "https://drive.google.com/file/d/10UJTh0YUpVk75H2KMIiZ61sDsC-woWl0/view?usp=sharing"
output_path = "temporal_ds.zip"  # Nombre del archivo descargado
extract_path = "/data/nisla"  # Carpeta donde se descomprimirán los archivos

# Descargar y descomprimir
download_file(file_url, output_path)
unzip_file(output_path, extract_path)

# Eliminar el archivo zip después de descomprimir (opcional)
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"Archivo {output_path} eliminado después de la descompresión.")