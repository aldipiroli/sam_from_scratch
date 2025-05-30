import os
import urllib.request
import tarfile

def download_pascal_voc_2012_dataset(destination_dir):
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    tar_filename = "VOCtrainval_11-May-2012.tar"
    tar_path = os.path.join(destination_dir, tar_filename)

    os.makedirs(destination_dir, exist_ok=True)

    if not os.path.exists(tar_path):
        print("Downloading Pascal VOC 2012...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print("Tar file already exists, skipping download.")

    extract_path = os.path.join(destination_dir, "VOCdevkit", "VOC2012")
    if not os.path.exists(extract_path):
        print("Extracting tar file...")
        with tarfile.open(tar_path, "r") as tar_ref:
            tar_ref.extractall(destination_dir)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    download_pascal_voc_2012_dataset("../data")
