import gdown
import zipfile
import os
import shutil
import argparse

def download_and_extract(url, destination):
    gdown.download(f"https://drive.google.com/uc?id={url}", destination, quiet=False)

    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(destination))
    os.remove(destination)

if __name__ == "__main__":
    path = "./data"
    if not os.path.exists(path):
        os.makedirs(path)

    file_urls = {
        '1': '11cU3DYDVtRPjIYLyT345y2L-wCOYagCK',
        '2': '1kXzF--BmVUNBdG2EHCF3jDb5QmYaj3EB',
        '3': '1IXgql4NqIrerbvvF3tDOu4ry0ibKHX83',
        '4': '1xtVfJWRaoVXwYjJFADQPRDukufgBHhIV',
        '5': '173efqT4u3h3ff45WIiIHFLQWoTukUPlz'
    }

    parser = argparse.ArgumentParser(description='Download and extract ZIP file from Google Drive.')
    parser.add_argument('options', 
                        help='Select one or more options for the URLs (e.g., "12" or "3"). \n'
                             '1: corresponds to the 2019a-smoke-full \n'
                             '2: corresponds to the AiForMankind \n'
                             '3: corresponds to the total Combine \n '
                             '4: corresponds to the SmokesFrames-2.4k\n '
                             '5: corresponds to the Wilfire_2023',
                        type=str)

    args = parser.parse_args()

    selected_options = args.options

    for option in selected_options:
        url_id = file_urls.get(option)
        if url_id:
            destination = os.path.join(path, f'file_{url_id}.zip')
            download_and_extract(url_id, destination)
        else:
            print(f"Invalid option '{option}'.")

    print("Download and extraction completed.")