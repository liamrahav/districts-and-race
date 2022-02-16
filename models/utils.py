DATASET_PATH = "dataset/preprocessed/dataset.pickle.bz2"

REDUCED_FEATURES = [
    "year",
    "totalPopRaceFile",
    "prcntUnemp",
    "prcntBA",
    "prcntHS",
    "gini",
    "CvxHullPT",
]

FEATURES = REDUCED_FEATURES + [
    "prcntBlack",
    "prcntHisp",
    "prcntAsian",
    "prcntWhiteAll",
]

LABELS = [
    "black",
    "hispanic",
]
