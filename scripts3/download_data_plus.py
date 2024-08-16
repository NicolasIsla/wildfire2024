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
    path = "./data"
    if not os.path.exists(path):
        os.makedirs(path)

    file_urls = {
        '1': {'id': '1TzlZXH7NvmhqMn4Sgf4_9c_j0Ky4hKMr', 'name': 'DS-71c1fd51-v2'},
        '2': {'id': '17aZEMghChHkYLjdi-yvMNsxxC6SdT6fw', 'name': 'FigLib'},
        '3': {'id': '1Zz13mnYVfWFm5PwO0n1P9WhuIPJg9H81', 'name': 'Nemo'}
    }

    parser = argparse.ArgumentParser(description='Download and extract files from Google Drive.')
    parser.add_argument('options', 
                        help='Select one or more options for the URLs (e.g., "12" or "3"). \n'
                             '1: corresponds to DS-71c1fd51-v2 \n'
                             '2: corresponds to FigLib \n'
                             '3: corresponds to Nemo',
                        type=str)

    args = parser.parse_args()

    selected_options = args.options

    for option in selected_options:
        file_info = file_urls.get(option)
        if file_info:
            url_id = file_info['id']
            file_name = file_info['name']
            file_type = 'zip'
            # create a folder with the name of the file
            path_folder = os.path.join(path, file_name)
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)
            destination = os.path.join(path_folder, f'{file_name}.{file_type}')
            download_and_extract(url_id, destination)
        else:
            print(f"Invalid option '{option}'.")

    print("Download and extraction completed.")



