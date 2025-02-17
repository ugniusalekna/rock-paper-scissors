import os
import zipfile
import requests
import inquirer


DATASETS = {
    "cifar10": "1PKd-2Ou5IwXBerbnHEl9DRlB80Aby0W_",
    "EMNIST": "1yUqBM-QCyjdpsrJQLCLtf1NxyipNStGo",
    "FER2013": "1pR7US6NuBRcQVFZme5FV6eBr5LrARz2Y",
    "QuickDraw": "13aA23PBAYZDDeGiJl6qFeryVkhQXN34A",
    "Download ALL": "ALL"
}


def download_file_from_google_drive(file_id, destination):
    URL = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    with requests.get(URL, stream=True) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(32768):
                f.write(chunk)


def extract_zip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(file_path)


def download_and_extract(dataset_name, file_id, data_dir):
    file_path = f"{data_dir}/{dataset_name}.zip"

    print(f"Downloading {dataset_name}...")
    download_file_from_google_drive(file_id, file_path)

    print(f"Extracting {dataset_name}...")
    extract_zip(file_path, data_dir)

    print(f"Dataset is ready in {data_dir}/{dataset_name}")


def main():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    questions = [
        inquirer.List(
            "dataset",
            message="Select a dataset to download:",
            choices=list(DATASETS.keys()),
        )
    ]
    answer = inquirer.prompt(questions)
    dataset_name = answer["dataset"]

    if dataset_name == "Download ALL":
        for name, file_id in DATASETS.items():
            if file_id != "ALL":
                download_and_extract(name, file_id, data_dir)
        print("All datasets have been downloaded and extracted.")
    else:
        download_and_extract(dataset_name, DATASETS[dataset_name], data_dir)


if __name__ == "__main__":
    main()