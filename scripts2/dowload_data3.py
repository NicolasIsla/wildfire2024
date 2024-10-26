import gdown
import zipfile
import os
import argparse

def download_and_extract(url, destination, cookies_path=None):
    try:
        # Si se proporcionan cookies, las usamos para la descarga
        if cookies_path:
            gdown.download(f"https://drive.google.com/uc?id={url}", destination, quiet=False, use_cookies=cookies_path)
        else:
            gdown.download(f"https://drive.google.com/uc?id={url}", destination, quiet=False)

        # Descomprime el archivo ZIP sin crear carpeta contenedora
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            temp_dir = os.path.join(os.path.dirname(destination), 'temp_extracted')
            zip_ref.extractall(temp_dir)

            # Mover los archivos al directorio destino
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    src = os.path.join(root, file)
                    dst = os.path.join(os.path.dirname(destination), file)
                    os.rename(src, dst)

            # Eliminar la carpeta temporal
            os.rmdir(temp_dir)

        # Eliminar el archivo ZIP
        os.remove(destination)

if __name__ == "__main__":
    path = "/data/nisla/Smoke50v3/DS/images"
    if not os.path.exists(path):
        os.makedirs(path)

    file_urls = {
        '1': {'id': '11cU3DYDVtRPjIYLyT345y2L-wCOYagCK', 'name': '2019a-smoke-full'},
        '2': {'id': '1kXzF--BmVUNBdG2EHCF3jDb5QmYaj3EB', 'name': 'AiForMankind'},
        # https://drive.google.com/file/d/1QPfjcLlHPhZ52I84KOm7Ar5BMSSIGq17/view?usp=drive_link
        '3': {'id': '1QPfjcLlHPhZ52I84KOm7Ar5BMSSIGq17', 'name': 'val'},
        # https://drive.google.com/file/d/1D4X0SfPiZjyadu2mEtU9YbKmQBid7-wT/view?usp=drive_link
        '4': {'id': '1D4X0SfPiZjyadu2mEtU9YbKmQBid7-wT', 'name': 'test'},
        
    }
    parser = argparse.ArgumentParser(description='Download and extract ZIP file from Google Drive.')
    parser.add_argument('options', 
                        help='Select one or more options for the URLs (e.g., "12" or "3"). \n'
                             '1: corresponds to the 2019a-smoke-full \n'
                             '2: corresponds to the AiForMankind \n'
                             '3: corresponds to the total Combine \n'
                             '4: corresponds to the SmokesFrames-2.4k\n'
                             '5: corresponds to the Wilfire_2023\n'
                             '6: corresponds to the Nemo\n'
                             '7: corresponds to the DS_08_V1\n'
                             '8: corresponds to the DS_08_V2',
                        type=str)
    parser.add_argument('--cookies', type=str, default=None, 
                        help='Path to cookies.txt file for authentication.')

    args = parser.parse_args()
    selected_options = args.options
    cookies_path = args.cookies

    for option in selected_options:
        file_info = file_urls.get(option)
        if file_info:
            url_id = file_info['id']
            file_name = file_info['name']
            # Crea una carpeta con el nombre del archivo
            path_folder = os.path.join(path, file_name)
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)
            destination = os.path.join(path_folder, f'{file_name}.zip')
            download_and_extract(url_id, destination, cookies_path)
        else:
            print(f"Invalid option '{option}'.")

    print("Download and extraction completed.")