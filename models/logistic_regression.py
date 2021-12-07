import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import DATASET_PATH, FEATURES, LABELS

if __name__ == '__main__':
    dataset = pd.read_pickle(DATASET_PATH)

    X = dataset[FEATURES].to_numpy()

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

    # We're ready to fit the model!
    clf = LogisticRegression(tol=1e-9, random_state=0, max_iter=1e6).fit(X, y)
    print(clf.score(X, y))
