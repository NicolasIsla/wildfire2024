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
        '6': {'id': '1zSRrONobhe2_CDVD4yayWCOLh-5KPRv5', 'name': 'Nemo'},
        '7': {'id': '1NJ7SsGZGlHP-a2MWurNZV3QojGxpbnem', 'name': 'DS_08_V1'},
        '8': {'id': '1TUuPNzBiFWhQ7EdgNRj8fs0IlwYLX1tj', 'name': 'DS_08_V2'},
        # https://drive.google.com/file/d/1p_2nQMZ2FbftULl9pEkgKAzygpefrpAU/view?usp=sharing
        '9': {'id': '1p_2nQMZ2FbftULl9pEkgKAzygpefrpAU', 'name': 'PyroVideo'},
        # https://drive.google.com/file/d/19Z0P0swSWuffqTuoVMdKfdmUPaz7zxR8/view?usp=drive_link
        '0': {'id': '19Z0P0swSWuffqTuoVMdKfdmUPaz7zxR8', 'name': 'PyroVideoEmpty'},
        # https://drive.google.com/file/d/1FRht00sPogpL-jKojYuUgXoLzljiR4zS/view?usp=sharing
        'a': {'id': '1FRht00sPogpL-jKojYuUgXoLzljiR4zS', 'name': 'videotest'},
        # https://drive.google.com/file/d/1PnzGMCz8ipCz0Mrvc1QLvaz4RjO6sfSf/view?usp=drive_link
        'b': {'id': '1PnzGMCz8ipCz0Mrvc1QLvaz4RjO6sfSf', 'name': 'NemoV2'},
        # https://drive.google.com/file/d/1CQm1yTZc9wv74BZmk8Qr_J9zSZ867XqH/view?usp=sharing
        'c': {'id': '1CQm1yTZc9wv74BZmk8Qr_J9zSZ867XqH', 'name': 'NemoV3'},
        # https://drive.google.com/file/d/1CLDHVSGIKZz9fZdV1TqbIPRfPSYvR3aA/view?usp=sharing
        'd': {'id': '1CLDHVSGIKZz9fZdV1TqbIPRfPSYvR3aA', 'name': 'Dfire'},
        # https://drive.google.com/file/d/1LuQKxIMtxOrWch-uxai7_FdMiuN0p5-f/view?usp=drive_link
        'e': {'id': '1LuQKxIMtxOrWch-uxai7_FdMiuN0p5-f', 'name': 'Smoke50'},
        # https://drive.google.com/file/d/1gCbeNLzWyWhuHw9DmoG_0er9Cz1vdwKg/view?usp=sharing
        'f': {'id': '1gCbeNLzWyWhuHw9DmoG_0er9Cz1vdwKg', 'name': 'Smoke50v2'},
        # https://drive.google.com/file/d/1KJkR06wFKcp57Pvd5E12E3y9uuAAIuxk/view?usp=drive_link
        'g': {'id': '1KJkR06wFKcp57Pvd5E12E3y9uuAAIuxk', 'name': 'Smoke50v3'},

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
                            '7: corresponds to the DS_08_V1\n '
                            '8: corresponds to the DS_08_V2',
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

