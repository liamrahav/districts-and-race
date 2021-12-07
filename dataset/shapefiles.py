"""
Download & unzip shapefiles from the United States Congressional District Shapefiles
dataset provided by UCLA. Downloads from the 93rd Congress onwards.

Running this module directly will download the shapefiles and clean them of
districts with boundaries (Washington D.C. for Congress 103 and onward)
"""
import atexit
import os
from pathlib import Path
from typing import Tuple
import urllib.request
from zipfile import ZipFile

from tqdm import trange
import fiona

BASE_URL = "https://cdmaps.polisci.ucla.edu/shp/districts{:03d}.zip"
BASE_OUTPUT_DIR = "dataset/raw/shapefiles/"
BASE_PROCESSED_DIR = "dataset/preprocessed/shapefiles"
CONGRESS_LOWER = 93 # inclusive
CONGRESS_UPPER = 114 # exclusive


def download_shapefiles(output_dir: str) -> None:
    for i in trange(CONGRESS_LOWER, CONGRESS_UPPER, desc="Downloading Shapefiles"):
        # yields a path to a temporary file, cleaned up at end of script
        path, _ = urllib.request.urlretrieve(BASE_URL.format(i))

        with ZipFile(path, "r") as zip_file:
            if output_dir[-1] != "/":
                output_dir += "/"
            directory = "{}{}".format(output_dir, i)
            Path(directory).mkdir(parents=True, exist_ok=True)
            zip_file.extractall(path=directory)

    urllib.request.urlcleanup()


def clean_shapefiles() -> None:
    # We need to clean DC from all Congresses >=103, since it is listed as a
    # district but has no boundaries. Without this step, scoring will error out.
    for cong_num in trange(CONGRESS_LOWER, CONGRESS_UPPER, desc="Cleaning Shapefiles"):
        shapefile_path = os.path.join(
            BASE_OUTPUT_DIR,
            str(cong_num) + "/districtShapes/",
            "districts{:03d}.shp".format(int(cong_num)),
        )

        output_path = os.path.join(
            BASE_PROCESSED_DIR,
            str(cong_num) + "/districtShapes"
        )
        Path(output_path).mkdir(parents=True, exist_ok=True)

        with fiona.open(shapefile_path) as source, fiona.open(
            output_path, "w", driver=source.driver, crs=source.crs, schema=source.schema
        ) as output:
            for element in source:
                # Ignore districts without geometry, verified to be only DC for
                # Congress No. 103 and above
                if element["geometry"] is not None:
                    output.write(element)


def fetch_scores_from(
    shapefile_path: str, score_names: Tuple[str] = ("CvxHullPT",)
) -> dict:
    """Returns a dict with the following format:
    {
        "STATENAME": {
            districtNum: {
                "scoreName": score
            }
        }
    }
    """

    result = {}
    with fiona.open(shapefile_path) as shapefile:
        for shape in shapefile:
            state = shape["properties"]["STATENAME"].upper()
            if state not in result.keys():
                result[state] = {}

            district = int(shape["properties"]["DISTRICT"])
            if district not in result[state].keys():
                result[state][district] = {}

            for score in score_names:
                try:
                    result[state][district][score] = shape["properties"][score]
                except KeyError:
                    print(
                        "Score Type {} not present for {} district {} in file {}".format(
                            score, state, district, shapefile_path
                        )
                    )

    return result


if __name__ == "__main__":
    atexit.register(urllib.request.urlcleanup)
    # download_shapefiles(BASE_OUTPUT_DIR)
    clean_shapefiles()