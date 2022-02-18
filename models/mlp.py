import os
import pprint

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from utils import get_dataset, get_test_set, REDUCED_FEATURES, FEATURES, LABELS, PRETRAINED_PREFIX


SAVE_MODEL_NAME = "mlp"
PARAM_GRID = {
    "hidden_layer_sizes": [
        (128,),
        (128, 128),
        (128, 64, 32),
        (64, 64, 64, 32, 16, 8, 4),
    ],
    "alpha": [1e-6, 1e-4, 1e-2],
    "learning_rate_init": [1e-1, 1e-3, 1e-5],
    "early_stopping": [True, False],
}


def perform_grid_search(dataset="full", verbose=True, save=True, load=True):
    filepath = "{}{}_{}.pkl".format(PRETRAINED_PREFIX, SAVE_MODEL_NAME, dataset)
    if load and os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except:
            pass

    X, y, cv = get_dataset(features=dataset)

    clf = MLPClassifier(random_state=None, shuffle=False, max_iter=100000)
    gs = GridSearchCV(
        clf,
        param_grid=PARAM_GRID,
        scoring=["accuracy", "balanced_accuracy", "f1_micro"],
        n_jobs=-1,
        cv=cv,
        refit="accuracy",
        verbose=2 if verbose else 0,
    ).fit(X, y)

    if verbose:
        pprint.pprint(gs.cv_results_)

    if save:
        joblib.dump(gs.best_estimator_, filepath)

    return gs.best_estimator_


if __name__ == "__main__":
    for dtype in ["full", "reduced"]:
        clf = perform_grid_search(dataset=dtype, load=False, save=False, verbose=False)
        print(clf.get_params())

        X, y = get_test_set(dtype)
        y_hat = clf.predict(X)
        f1 = f1_score(y_true=y, y_pred=y_hat, average='micro')
        print("{} penalty\tAcc: {:.4f}\tF1: {:.4f}".format(dtype, clf.score(X, y), f1))
