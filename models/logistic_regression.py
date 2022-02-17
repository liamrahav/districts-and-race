from matplotlib import pyplot as plt
import os
import pprint

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import normalize

from utils import get_dataset, set_plot_defaults, REDUCED_FEATURES, FEATURES, LABELS, PRETRAINED_PREFIX


SAVE_MODEL_NAME = "logistic_regression"


def perform_grid_search(
    dataset="full", verbose=True, save=True, load=True, penalty="l2"
):
    # Note that the Logistic Regression search has the additional "penalty" parameter
    # This can be either L1 or L2 (L1 generally enforcing sparsity on the learned parameters)
    filepath = "{}{}_{}_{}.pkl".format(
        PRETRAINED_PREFIX, SAVE_MODEL_NAME, penalty, dataset
    )
    if load and os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except:
            pass

    X, y, cv = get_dataset(features=dataset)

    # Use built in Logistic Regression Cross Validation with our custom groups
    # This only supports a single score in the printout, however
    gs = LogisticRegressionCV(
        scoring="f1_micro",
        n_jobs=-1,
        cv=cv,
        max_iter=1e7,
        refit=True,
        penalty=penalty,
        solver="liblinear",
        verbose=1 if verbose else 0,
    ).fit(X, y)

    if save:
        joblib.dump(gs, filepath)

    return gs


if __name__ == "__main__":
    set_plot_defaults()

    for i, dtype in enumerate(["full", "reduced"]):
        for j, penalty in enumerate(["l1", "l2"]):
            clf = perform_grid_search(dataset=dtype, verbose=False, penalty=penalty)

            X, y, cv = get_dataset(features=dtype)
            print("{}, {} penalty\t{}".format(dtype, penalty, clf.score(X, y)))

            featnames = FEATURES if dtype == "full" else REDUCED_FEATURES
            fig, axs = plt.subplots(1, 3, figsize=(30, 10))
            for ax in axs.flat:
                ax.set(xlabel="feature", ylabel="weight")

            # Graph normalized weights for all classes
            for k, cname in enumerate(["white", "black", "hispanic"]):
                sns.barplot(
                    x=featnames,
                    y=clf.coef_[k] / np.linalg.norm(clf.coef_[k]),
                    palette="deep",
                    ax=axs[k],
                )
                axs[k].set_title(
                    "{} Dataset, {} Penalty — {}".format(
                        dtype.title(), penalty.upper(), cname
                    )
                )
                plt.setp(
                    axs[k].get_xticklabels(), rotation=30, horizontalalignment="right"
                )

            plt.savefig('figures/lr_weights_{}_{}.png'.format(penalty, dtype))
