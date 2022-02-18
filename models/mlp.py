import os
import pprint

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from utils import get_dataset, REDUCED_FEATURES, FEATURES, LABELS, PRETRAINED_PREFIX


SAVE_MODEL_NAME = "mlp"
PARAM_GRID = {
    "hidden_layer_sizes": [
        (200, 200), (200, 200, 200), (200, 200, 200, 200),
        (400, 400), (400, 400, 400), (400, 400, 400, 400),
        (800, 800), (800, 800, 800), (800, 800, 800, 800),
    ],
    "alpha": [1e-6, 1e-4, 1e-2]
}


def perform_grid_search(dataset="full", verbose=True, save=True, load=True):
    filepath = "{}{}_{}.pkl".format(PRETRAINED_PREFIX, SAVE_MODEL_NAME, dataset)
    if load and os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except:
            pass

    X, y, cv = get_dataset(features=dataset)

    clf = MLPClassifier(random_state=0, solver='adam')
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
    clf = perform_grid_search(dataset="full", load=False)

    X, y, _ = get_dataset(features='full')
    print(clf.score(X, y))
