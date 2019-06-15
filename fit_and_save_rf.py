"""
Create a model based on a list of features

You can call it like so:

    python -m newcomb.fit_and_save_rf newcomb-server/newcomb-data.csv gender age payoff1 payoff2
"""
import argparse
import json
import pdb
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from . import utils

PICKLE_FILE = "newcomb-model.pickle"


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
    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("datafile", help="the CSV file")
    parser.add_argument("features", nargs="+", help="the feature names, without payoffRatio")
    args = parser.parse_args()

    data = pd.read_csv(args.datafile)

    # Fit and save model
    model = fit_test_rf(data, args.features)

    # Save model
    with open(PICKLE_FILE, "wb") as pickle_file:
        pickle.dump(model, pickle_file)

    print("Please input a JSON list of features:")
    for line in sys.stdin:
        values = json.loads(line)
        print(model.predict([values]))
