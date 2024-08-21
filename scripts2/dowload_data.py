import gdown
import zipfile
import os
import argparse

def download_and_extract(url, destination):
    gdown.download(f"https://drive.google.com/uc?id={url}", destination, quiet=False)

    with zipfile.ZipFile(destination, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(destination))
    os.remove(destination)

if __name__ == "__main__":
    path = "/data/nisla"
    if not os.path.exists(path):
        os.makedirs(path)

    file_urls = {
        '1': {'id': '11cU3DYDVtRPjIYLyT345y2L-wCOYagCK', 'name': '2019a-smoke-full'},
        '2': {'id': '1kXzF--BmVUNBdG2EHCF3jDb5QmYaj3EB', 'name': 'AiForMankind'},
        '3': {'id': '1IXgql4NqIrerbvvF3tDOu4ry0ibKHX83', 'name': 'total_Combine'},
        '4': {'id': '1xtVfJWRaoVXwYjJFADQPRDukufgBHhIV', 'name': 'SmokesFrames-2.4k'},
        '5': {'id': '173efqT4u3h3ff45WIiIHFLQWoTukUPlz', 'name': 'Wilfire_2023'},
        '6': {'id': '1Zz13mnYVfWFm5PwO0n1P9WhuIPJg9H81', 'name': 'Nemo'},
        '7': {'id': '', 'name': 'DS-71'},
        '8': {'id': '', 'name': 'PyroNear_2024'},
        '9': {'id': '17aZEMghChHkYLjdi-yvMNsxxC6SdT6fw', 'name': 'FigLib'},


    }

    parser = argparse.ArgumentParser(description='Download and extract ZIP file from Google Drive.')
    parser.add_argument('options', 
                        help='Select one or more options for the URLs (e.g., "12" or "3"). \n'
                             '1: corresponds to the 2019a-smoke-full \n'
                             '2: corresponds to the AiForMankind \n'
                             '3: corresponds to the total Combine \n '
                             '4: corresponds to the SmokesFrames-2.4k\n '
                             '5: corresponds to the Wilfire_2023\n '
                            '6: corresponds to the Nemo\n '
                            '7: corresponds to the DS-71\n '
                            '8: corresponds to the PyroNear\n'
                            '9: corresponds to the FigLib',
                        type=str)

    args = parser.parse_args()

    selected_options = args.options

    for option in selected_options:
        file_info = file_urls.get(option)
        if file_info:
            url_id = file_info['id']
            file_name = file_info['name']
            # create a folder with the name of the file
            path_folder = os.path.join(path, file_name)
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)
            destination = os.path.join(path_folder, f'{file_name}.zip')
            download_and_extract(url_id, destination)
        else:
            print(f"Invalid option '{option}'.")

    print("Download and extraction completed.")
