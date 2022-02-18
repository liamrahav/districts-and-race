import os
import matplotlib.pyplot as plt
import pprint

import joblib
import numpy as np
import pandas as pd
import pydotplus
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

from utils import get_dataset, REDUCED_FEATURES, FEATURES, LABELS, PRETRAINED_PREFIX

SAVE_MODEL_NAME = "dt"
PARAM_GRID = {"max_depth": [2, 4, 6, 10, 15]}


def perform_grid_search(dataset="full", verbose=True, save=True, load=True):
    filepath = "{}{}_{}.pkl".format(PRETRAINED_PREFIX, SAVE_MODEL_NAME, dataset)
    if load and os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except:
            pass

    X, y, cv = get_dataset(features=dataset)

    clf = DecisionTreeClassifier(random_state=0)
    gs = GridSearchCV(
        clf,
        param_grid=PARAM_GRID,
        scoring=["accuracy", "balanced_accuracy", "f1_micro"],
        n_jobs=-1,
        cv=cv,
        refit="f1_micro",
        verbose=2 if verbose else 0,
    ).fit(X, y)

    if verbose:
        pprint.pprint(gs.cv_results_)

    if save:
        joblib.dump(gs.best_estimator_, filepath)

    return gs.best_estimator_


if __name__ == "__main__":
    clf = perform_grid_search(dataset='full')
    dot_file = "figures/tree_full_dataset.dot"
    export_graphviz(
        clf,
        dot_file,
        rounded=True,
        filled=True,
        class_names=["other"] + LABELS,
        feature_names=FEATURES,
    )
    pydotplus.graph_from_dot_file(dot_file).write_png(dot_file.replace(".dot", ".png"))
