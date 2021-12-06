"""
Download & unzip shapefiles from the United States Congressional District Shapefiles
dataset provided by UCLA. Downloads from the 93rd Congress onwards
"""
import atexit
from pathlib import Path
from typing import Tuple
import urllib.request
from zipfile import ZipFile

from tqdm import trange
import fiona

BASE_URL = "https://cdmaps.polisci.ucla.edu/shp/districts{:03d}.zip"
BASE_OUTPUT_DIR = "dataset/raw/shapefiles/"


def download_shapefiles(output_dir: str) -> None:
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


def fetch_scores_from(shapefile_path, score_names: Tuple[str] = ('CvxHullPT',)) -> dict:
    '''Returns a dict with the following format:
    {
        "stateName": {
            districtNum: {
                "scoreName": score
            }
        }
    }
    '''

    result = {}
    with fiona.open(shapefile_path) as shapefile:
        for shape in shapefile:
            state = shape['properties']['STATENAME']
            if state not in result.keys():
                result[state] = {}

            district = int(shape['properties']['DISTRICT'])
            if district not in result[state].keys():
                result[state][district] = {}

            for score in score_names:
                result[state][district][score] = shape['properties'][score]

    return result


if __name__ == "__main__":
    atexit.register(urllib.request.urlcleanup)
    download_shapefiles(BASE_OUTPUT_DIR)
