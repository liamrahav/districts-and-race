import json
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pydotplus
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV

from utils import DATASET_PATH, REDUCED_FEATURES, FEATURES, LABELS

if __name__ == '__main__':
    dataset = pd.read_pickle(DATASET_PATH)

    X = dataset[REDUCED_FEATURES].to_numpy()

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

    # We're ready to fit the model!
    clf = LogisticRegression(tol=1e-9, random_state=0, max_iter=1e6)
    clf = DecisionTreeClassifier(random_state=0)
    gs = GridSearchCV(clf, param_grid={'max_depth': [4,6,10,40]}, scoring=[
        'accuracy', 'balanced_accuracy', 'f1_micro'
    ], n_jobs=-1, cv=cv, refit='f1_micro', verbose=2).fit(X, y)

    import pprint
    pprint.pprint(gs.cv_results_)

    # From prior run we know that depth = 6 is best performing (and not too deep to visualize)
    clf = DecisionTreeClassifier(random_state=0, max_depth=6).fit(X, y)
    dot_file = 'figures/tree_reduced_dataset.dot'
    export_graphviz(clf, dot_file, rounded=True, filled=True, class_names=['white'] + LABELS, feature_names=REDUCED_FEATURES)
    pydotplus.graph_from_dot_file(dot_file).write_png(dot_file.replace('.dot', '.png'))
