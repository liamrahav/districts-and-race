"""
This module contains functions to preprocess the raw datasets to a single format that
will be used to train a machine learning model.

NOTE: MAKE SURE YOU HAVE DOWNLOADED & SCORED SHAPEFILES BEFORE RUNNING THIS SCRIPT.
"""
import argparse
import codecs
import csv
import math
import os
from typing import Tuple

import numpy as np
import pandas as pd

from dataset import utils
from shapefiles import fetch_scores_from


UP_TO_YEAR = 2012
WINNERS_DATASET = "dataset/raw/1976-2020-house.csv"
HCLDD_DATASET = "dataset/raw/allCongressDataPublishV2.csv"
HCLDD_FEATURES = [
    "totalPopRaceFile",
    "prcntUnemp",
    "prcntBA",
    "prcntHS",
    # "meanIncome",
    "gini",
    "prcntBlack",
    "prcntHisp",
    "prcntAsian",
    "prcntWhiteAll",
]
SHAPEFILES_BASEDIR = "dataset/preprocessed/shapefiles"
OUTPUT_DATASET = "dataset/preprocessed/dataset.pickle.bz2"


def _impute_winners(result_dict: dict) -> dict:
    did_win = [False] * len(result_dict["year"])
    winner_count = 0
    group_count = 0

    df = pd.DataFrame.from_dict(result_dict)

    # We want to assign one winner per "group", where a group is a given year, state, and district
    # AKA each individual House race
    cur_group = None
    base_index = 0
    votes = []
    candidates = []
    for i, row in df.iterrows():
        row_group = (row["year"], row["state"], row["district"])

        # current group is initially null
        if not cur_group:
            cur_group = row_group

        # If still in same group, just add candidate names and vote totals to current
        if row_group == cur_group:
            votes.append(int(row["candidatevotes"]))
            candidates.append(row["candidate"])

        # If we change groups, need to pick winner and reset currents
        if row_group != cur_group or i == len(df.index) - 1:
            winner = np.argmax(votes)

            did_win[base_index + winner] = True
            winner_count += 1

            cur_group = row_group
            group_count += 1

            base_index += len(votes)
            votes = [int(row["candidatevotes"])]
            candidates = [row["candidate"]]

    result_dict["did_win"] = did_win

    # double check that the number of winners exactly equals the number of unique races
    assert winner_count == group_count
    return result_dict


def _load_winners(winner_path: str) -> dict:
    el_res_dict = {}
    to_ignore = []

    # Took some digging to find out encoding of this data is iso-8859-1
    with codecs.open(winner_path, mode="r", encoding="iso-8859-1") as f:
        csv_data = csv.DictReader(f)
        for field in csv_data.fieldnames:
            el_res_dict[field] = []

        for row in csv_data.reader:
            # We skip duplicate candidate entries; this occurs when there are
            # "Fusion" tickets with one person running on behalf of multiple parties.
            # Since this is a small number of entries, we assume picking the first listed
            # party is OK. This can be revisited in the future if needed.
            if (row[0], row[2], row[7], row[11]) in to_ignore:
                # This is year, state, district, candidate name
                continue

            if row[-1].lower() == "true":
                to_ignore.append((row[0], row[2], row[7], row[11]))

            for i, item in enumerate(row):
                el_res_dict[csv_data.fieldnames[i]].append(item)

    return el_res_dict


def _load_hcldd(hcldd_path: str) -> dict:
    hcldd = {}
    with open(hcldd_path, mode="r") as f:
        csv_data = csv.DictReader(f)
        for field in csv_data.fieldnames:
            hcldd[field] = []

        for row in csv_data.reader:
            for i, item in enumerate(row):
                hcldd[csv_data.fieldnames[i]].append(item)

    # We manually clean the last 3 rows since they are spam/placeholder
    hcldd = {k: hcldd[k][:-3] for k in hcldd.keys()}

    # Next we transform `statDist` and `congNum` to state, district, and year
    # This will allow us to merge with the MIT Election Lab winnersd dataset
    hcldd["state_po"] = []
    hcldd["district"] = []
    hcldd["year"] = []

    for i in range(len(hcldd["stateDist"])):
        state_po, district = hcldd["stateDist"][i].split(".")
        hcldd["state_po"].append(state_po)
        hcldd["district"].append(district)

        year = ((int(hcldd["congNum"][i]) - 93) * 2) + 1972
        hcldd["year"].append(year)

    del hcldd["stateDist"]

    return hcldd


def _merge_compactness_scores(
    dataset: pd.DataFrame,
    shapefiles_dir: str,
    scores: Tuple[str] = ('CvxHullPT',)
) -> pd.DataFrame:
    for score_name in scores:
        dataset[score_name] = [0] * len(dataset.index)

    all_scores = {}
    for cong_num in dataset.congNum.unique():
        if math.isnan(float(cong_num)):
            continue

        shapefile_path = os.path.join(
            shapefiles_dir,
            str(cong_num) + "/districtShapes/districtShapes.shp",
        )
        all_scores[cong_num] = fetch_scores_from(shapefile_path, scores)

    for cong_num in all_scores:
        for state_name in all_scores[cong_num]:
            for district in all_scores[cong_num][state_name]:
                for score_name, score in all_scores[cong_num][state_name][district].items():
                    dataset.loc[
                        (dataset['congNum'] == str(cong_num)) &
                        (dataset['state'] == state_name) &
                        (dataset['district'] == str(district)),
                        score_name
                    ] = score

    return dataset


def build_dataset(
    winners_path: str,
    hcldd_path: str,
    shapefiles_dir: str,
) -> pd.DataFrame:
    results_dict = _impute_winners(_load_winners(winners_path))

    # go from dictionary of all results to dataframe of winners only
    winners_df = pd.DataFrame.from_dict(results_dict)
    winners_df = winners_df[winners_df["did_win"] == True].reset_index()

    # only maintain needed columns
    winners_df = winners_df[["candidate", "year", "state", "state_po", "district"]]

    # clean dataset ?????remove underscores from names and delete misc. values
    # from analysis there are ~50 entries not attributes to an individual
    winners_df["candidate"] = winners_df["candidate"].apply(utils.clean_underscores)
    no_good = [x for x in winners_df["candidate"].tolist() if " " not in x.strip()]
    winners_df = winners_df[~winners_df["candidate"].isin(set(no_good))]

    hcldd = _load_hcldd(hcldd_path)

    # Merge race labels with winners dataframe
    race_labels = {
        k: hcldd[k]
        # Maintain congNum for ease when merging compactness scores
        for k in ("year", "congNum", "state_po", "district", "black", "hispanic")
        if k in hcldd
    }
    race_df = pd.DataFrame.from_dict(race_labels)
    winners_df["year"] = pd.to_numeric(winners_df["year"])
    labels_df = pd.merge(
        winners_df, race_df, how="left", on=["state_po", "district", "year"]
    )

    # Merge features into dataframe
    feature_dict = {
        k: hcldd[k]
        for k in HCLDD_FEATURES + ["state_po", "district", "year"]
        if k in hcldd
    }

    dataset = pd.merge(
        labels_df[labels_df["year"] <= UP_TO_YEAR],
        pd.DataFrame.from_dict(feature_dict),
        how="left",
        on=["state_po", "district", "year"],
    )

    dataset = _merge_compactness_scores(dataset, shapefiles_dir)

    to_numbers = HCLDD_FEATURES + ["black", "hispanic", "CvxHullPT"]
    dataset[to_numbers] = dataset[to_numbers].apply(pd.to_numeric, errors="coerce")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.description = (
        "Preprocesses the raw data and outputs a pickle of a Pandas dataframe"
        " containing the dataset to be used for model training and inference."
        "\nNOTE: MAKE SURE YOU HAVE DOWNLOADED & SCORED SHAPEFILES BEFORE RUNNING THIS SCRIPT."
    )
    parser.add_argument("--hcldd-path", "-f", default=HCLDD_DATASET)
    parser.add_argument("--winners-path", "-w", default=WINNERS_DATASET)
    parser.add_argument("--shapefiles-path", "-s", default=SHAPEFILES_BASEDIR)
    parser.add_argument("--output-path", "-o", default=OUTPUT_DATASET)
    args = parser.parse_args()

    dataset = build_dataset(args.winners_path, args.hcldd_path, args.shapefiles_path)
    dataset.to_pickle(args.output_path)
    print("Done!\nPath: {}".format(args.output_path))
