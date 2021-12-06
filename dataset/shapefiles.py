"""
Download & unzip shapefiles from the United States Congressional District Shapefiles
dataset provided by UCLA. Downloads from the 93rd Congress onwards
"""
import atexit
from pathlib import Path
import urllib.request
from zipfile import ZipFile

import mander
from tqdm import trange

BASE_URL = "https://cdmaps.polisci.ucla.edu/shp/districts{:03d}.zip"
BASE_OUTPUT_DIR = "dataset/raw/shapefiles/"


def download_shapefiles(output_dir) -> None:
    for i in trange(93, 114):
        # yields a path to a temporary file, cleaned up at end of script
        path, _ = urllib.request.urlretrieve(BASE_URL.format(i))

        with ZipFile(path, "r") as zip_file:
            if output_dir[-1] != "/":
                output_dir += "/"
            directory = "{}{}".format(output_dir, i)
            Path(directory).mkdir(parents=True, exist_ok=True)
            zip_file.extractall(path=directory)

    urllib.request.urlcleanup()


if __name__ == "__main__":
    atexit.register(urllib.request.urlcleanup)
    download_shapefiles(BASE_OUTPUT_DIR)
