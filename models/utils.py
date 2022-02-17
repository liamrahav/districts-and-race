import pandas as pd
import numpy as np

DATASET_PATH = "dataset/preprocessed/dataset.pickle.bz2"
PRETRAINED_PREFIX = 'pretrained/'

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


def get_dataset(features='full'):
    if features not in ('full', 'reduced'):
        raise ValueError('features must be "full" or "reduced".')

    featnames = FEATURES if features == 'full' else REDUCED_FEATURES

    dataset = pd.read_pickle(DATASET_PATH)

    X = dataset[featnames].to_numpy()

    # We know that black is the first column and hispanic is second
    # To get labels, simply turn all hispanic from 1 to 2 via multiplication
    # Then merge all the labels. This yields: other = 0, black = 1, hispanic = 2
    y = dataset[LABELS].to_numpy()
    y[:,1] *= 2
    y = y.sum(axis=1)

    # We notice 4 columns where race label is NaN. Let's just strike those out
    to_del = (np.argwhere(np.isnan(y)).T)[0]
    X = np.delete(X, to_del, axis=0)
    y = np.delete(y, to_del, axis=0)
    assert X.shape[0] == y.shape[0]

    # Set up initial split as follows:
    # TRAIN: 1976               TEST: 1978
    # TRAIN: 1976, 1978         TEST: 1980
    # ...
    # TRAIN: 1976, ..., 2010    TEST: 2012
    groups = []
    for year in list(dataset.year.unique()):
        groups.append(np.where(X[:,0] == year)[0].tolist())

    cv = []
    for i in range(len(groups) - 1):
        subgroup = []
        for j in range(i + 1):
            subgroup += list(groups[j])

        cv.append((subgroup, groups[i + 1]))

    return X, y, cv
