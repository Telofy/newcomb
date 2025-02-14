import numpy as np
import pandas as pd
from sklearn.metrics import recall_score


def split_dataset_by_study(data, feature_names, excluded_study_labels=(20,)):
    """

    :param data: Pandas df containing full newcomb data.
    :param excluded_study_labels:
    :return:
    """
    PAYOFF_DICT = {
        1.0: (3, 0.05),
        2.0: (2.55, 0.45),
        3.0: (2.25, 0.75),
        12.0: (3, 0.50),
        13.0: (2.5, 0.25),
        14.0: (2.5, 0.25),
        15.0: (4, 0.5),
        16.0: (4, 0.5),
        17.0: (4, 0.5),
        18.0: (4, 0.5),
        19.0: (2.23, 0.28),
        20.0: (2.23, 0.11),
        21.0: (2.23, 0.11),
        22.0: (2, 0.1),
    }  # Payoffs for (box 1, box 2) in each study

    # Remove columns not needed for analysis
    cols_to_remove = [
        "StartDate",
        "EndDate",
        "Platform",
        "IPAddress",
        "workerId",
        "Duration__in_seconds_",
        "RecordedDate",
        "newcomb_explanation",
        "newcomb_two_explain",
        "newcomb_confidence",
        "comprehension",
        "believability_prediction",
        "believability_scenario",
        "believability_1",
        "believability_2",
        "believability_3",
        "knowing_prediction",
        "decoding",
        "feedback",
        "fair",
        "long",
        "hard",
    ]
    # Also drop cols that aren't provided in feature_names
    if feature_names is not None:
        for col in data.columns:
            if col not in ["Study", "newcomb_combined"] and col not in feature_names:
                cols_to_remove.append(col)
    data.drop(labels=cols_to_remove, axis=1, inplace=True, errors="ignore")

    if "ethnicity" in data.columns:
        # Convert ethnicity coding from numeric to categorical
        data["ethnicity"] = data.ethnicity.astype("category")

    # Drop empty columns, then take complete cases (null values are coded as ' ',
    # need to change to nan)
    data.replace(" ", np.nan, inplace=True)
    data.dropna(axis=1, how="all", inplace=True)
    data.dropna(axis=0, how="any", inplace=True)
    data = data.applymap(float)

    # Create separate dataframes for each study
    dataframes_for_each_study = {}
    for study_number in data.Study.unique():
        if not np.isnan(study_number) and study_number not in excluded_study_labels:
            data_for_study = data[data["Study"] == study_number]

            data_for_study = data_for_study[data_for_study["newcomb_combined"] != 0.0]
            print("Study {} data size {}".format(study_number, data_for_study.shape))

            if data_for_study.shape[0] > 0:
                X_for_study = data_for_study.drop(labels=["newcomb_combined", "Study"], axis=1)
                if "payoff1" in feature_names and "payoff2" in feature_names:
                    X_for_study["payoff1"] = PAYOFF_DICT[study_number][0]
                    X_for_study["payoff2"] = PAYOFF_DICT[study_number][1]
                    X_for_study["payoffRatio"] = X_for_study.payoff1 / X_for_study.payoff2
                y_for_study = data_for_study.newcomb_combined
                dataframes_for_each_study[study_number] = (X_for_study, y_for_study)
    return dataframes_for_each_study


def balanced_accuracy_score(y_true, y_pred):
    """
    Assuming y binary.

    :param y_pred:
    :param y_true:
    :return:
    """
    y_vals = np.unique(y_true)
    recall_1 = recall_score(y_true, y_pred, pos_label=y_vals[0])
    recall_2 = recall_score(y_true, y_pred, pos_label=y_vals[1])
    return (recall_1 + recall_2) / 2


def bpa_scorer(estimator, X, y):
    """

    :param estimator:
    :param X:
    :param y:
    :return:
    """
    y_pred = estimator.predict(X)
    return balanced_accuracy_score(y, y_pred)


def balanced_accuracy_for_optimal_threshold(y_true, phat):
    best_accuracy = 0.0
    best_threshold = None
    for threshold in phat:
        y_pred = (phat > threshold) + 1
        bpa = balanced_accuracy_score(y_true, y_pred)
        if bpa > best_accuracy:
            best_threshold = threshold
            best_accuracy = bpa
    return best_accuracy, best_threshold
