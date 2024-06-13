import requests
import zipfile
import os


def download_tiny_imagenet(destination_folder):
    # URL for Tiny ImageNet dataset
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    # Download the dataset
    response = requests.get(url, stream=True)
    zip_path = os.path.join(destination_folder, "tiny-imagenet-200.zip")

    with open(zip_path, 'wb') as zip_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):  # Download in 1MB chunks
            if chunk:
                zip_file.write(chunk)

    # Extract the dataset
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

    # Optionally remove the ZIP file after extraction
    os.remove(zip_path)

    print(f"Tiny ImageNet has been downloaded and extracted to {destination_folder}")


if __name__ == "__main__":
    destination = "C:\\path\\to\\your\\destination\\folder"  # Modify this to your desired location
    download_tiny_imagenet(destination)
