import pdb
import pickle as pkl
import random

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from . import utils



def fit_test_rf(data, feature_names):
    random.seed(1)
    dataframes_for_each_study = utils.split_dataset_by_study(
        data, feature_names, excluded_study_labels=(20,)
    )

    # Merge dataframes
    X = pd.DataFrame({f: [] for f in feature_names})
    y = np.zeros(0)
    for X_, y_ in dataframes_for_each_study.values():
        X = pd.concat([X, X_])
        y = np.append(y, y_)

    # Fit model on merged dataset
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    return clf


if __name__ == "__main__":
    feature_names = ["gender", "age", "payoff1", "payoff2"]
    model_fname = "test_model.sav"

    # Fit and save model
    fit_and_save_test_rf(model_fname, feature_names)

    # Load model and data to get test datapoint
    test_model = pkl.load(open("test_model.sav", "rb"))
    data = pd.read_csv("newcomb-data.csv")
    x_test = [[1, 30, 20, 3, 0.5]]

    # Run model on test point
    print(test_model.predict(x_test))
