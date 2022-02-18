import os
import matplotlib.pyplot as plt
import pprint

import joblib
import numpy as np
import pandas as pd
import pydotplus
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from utils import get_dataset, get_test_set, REDUCED_FEATURES, FEATURES, LABELS, PRETRAINED_PREFIX

SAVE_MODEL_NAME = "dt"
PARAM_GRID = {"max_depth": [2, 3, 6, 10, 15]}


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
    for dtype in ["full", "reduced"]:
        clf = perform_grid_search(dataset=dtype, load=False, verbose=False)
        print(clf.get_params())

        X, y = get_test_set(dtype)
        y_hat = clf.predict(X)
        f1 = f1_score(y_true=y, y_pred=y_hat, average='weighted')
        print("{} penalty\tAcc: {:.4f}\tF1: {:.4f}".format(dtype, clf.score(X, y), f1))

        dot_file = "figures/tree_{}_dataset.dot".format(dtype)
        export_graphviz(
            clf,
            dot_file,
            rounded=True,
            filled=True,
            class_names=["other"] + LABELS,
            feature_names=FEATURES if dtype == 'full' else REDUCED_FEATURES,
        )
        pydotplus.graph_from_dot_file(dot_file).write_png(dot_file.replace(".dot", ".png"))
