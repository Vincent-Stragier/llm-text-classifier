"""A simple script to download the datasets from the web"""

import os
import requests

from constants import (
    URL_DATASET_A,
    URL_DATASET_B,
    PATH_DATASET_A,
    PATH_DATASET_B,
)
from utils import ensure_path_exists


def download_file(url, filename):
    """Download a file from the web"""
    ensure_path_exists(os.path.dirname(filename))

    response = requests.get(url, timeout=5)
    response.raise_for_status()

    with open(filename, "wb") as file:
        file.write(response.content)


if __name__ == "__main__":
    download_file(URL_DATASET_A, PATH_DATASET_A)
    download_file(URL_DATASET_B, PATH_DATASET_B)
    print("Files downloaded successfully")
