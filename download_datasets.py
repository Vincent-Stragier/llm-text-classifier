"""A simple script to download the datasets from the web"""
import requests

URL_A = "https://raw.githubusercontent.com/Vincent-Stragier/prompt_based_dataset_generation/ac84a1a640a209bd2d1047ce1bf9cae62374e3d5/datasets/results_prompts_0.json"
URL_B = "https://raw.githubusercontent.com/Vincent-Stragier/prompt_based_dataset_generation/ac84a1a640a209bd2d1047ce1bf9cae62374e3d5/datasets/results_prompts_1.json"


def download_file(url, filename):
    """Download a file from the web"""
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    with open(filename, 'wb') as file:
        file.write(response.content)


if __name__ == "__main__":
    download_file(URL_A, "dataset_a.json")
    download_file(URL_B, "dataset_b.json")
    print("Files downloaded successfully")
