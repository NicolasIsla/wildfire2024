import requests
import os

def download_from_gdrive(file_id, destination):
    # URL base para descargar desde Google Drive
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()

    # Primer intento de obtener el archivo
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    # Si hay un token de confirmación, realiza otro intento
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Guarda el contenido en el archivo de destino
    save_response_content(response, destination)

def get_confirm_token(response):
    """Extrae el token de confirmación del contenido HTML."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Guarda el contenido del archivo en el destino indicado."""
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filtro para omitir respuestas vacías
                f.write(chunk)

if __name__ == "__main__":
    # ID del archivo en Google Drive
    file_id = "1KJkR06wFKcp57Pvd5E12E3y9uuAAIuxk"
    
    # Ruta de destino donde se guardará el archivo descargado
    destination = "/data/nisla/Smoke50v3.zip"

    # Descargar el archivo
    print("Iniciando la descarga...")
    download_from_gdrive(file_id, destination)
    print(f"Descarga completada: {destination}")
