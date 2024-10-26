import gdown
import zipfile
import os
import argparse

def download_and_extract(url, destination, cookies_path=None):
    try:
        # Descarga usando cookies si se proporcionan
        if cookies_path:
            gdown.download(f"https://drive.google.com/uc?id={url}", destination, quiet=False, use_cookies=cookies_path)
        else:
            gdown.download(f"https://drive.google.com/uc?id={url}", destination, quiet=False)

        # Extrae el contenido del ZIP en el mismo directorio del archivo ZIP
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(destination))
        os.remove(destination)

    except Exception as e:
        print(f"Error downloading or extracting: {e}")

if __name__ == "__main__":
    path =  "/data/nisla/Smoke50v3/DS/labels"
    if not os.path.exists(path):
        os.makedirs(path)

    file_urls = {
        # https://drive.google.com/file/d/1BzrKREedNtPsZukliB1Sl64e9e6txinQ/view?usp=drive_link
        '1': {'id': '1BzrKREedNtPsZukliB1Sl64e9e6txinQ', 'name': 'labels'},
        # Agrega más opciones según sea necesario...
    }

    parser = argparse.ArgumentParser(description='Download and extract ZIP file from Google Drive.')
    parser.add_argument('options', 
                        help='Select one or more options for the URLs (e.g., "12" or "3").', 
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
            destination = os.path.join(path, f'{file_name}.zip')
            download_and_extract(url_id, destination, cookies_path)
        else:
            print(f"Invalid option '{option}'.")

    print("Download and extraction completed.")
