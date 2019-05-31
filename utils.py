import numpy as np
import pandas as pd


def split_dataset_by_study(data, exluded_study_labels=(20,)):
  """

  :param data: Pandas df containing full newcomb data.
  :param exluded_study_labels:
  :return:
  """
  PAYOFF_DICT = {1.0: (3, 0.05), 2.0: (2.55, 0.45), 3.0: (2.25, 0.75), 12.0: (3, 0.50), 13.0: (2.5, 0.25),
                 14.0: (2.5, 0.25), 15.0: (4, 0.5), 16.0: (4, 0.5), 17.0: (4, 0.5), 18.0: (4, 0.5), 19.0: (2.23, 0.28),
                 20.0: (2.23, 0.11), 21.0: (2.23, 0.11), 22.0: (2, 0.1)}  # Payoffs for (box 1, box 2) in each study

  # Remove columns not needed for analysis
  cols_to_remove = ["StartDate", "EndDate", "Platform", "IPAddress", "workerId", "Duration__in_seconds_",
                    "RecordedDate", "newcomb_explanation", "newcomb_two_explain", "newcomb_confidence",
                    "comprehension", "believability_prediction", "believability_scenario", "believability_1",
                    "believability_2", "believability_3", "knowing_prediction", "decoding", "feedback", "fair",
                    "long", "hard"]
  data.drop(labels=cols_to_remove, axis=1, inplace=True)

  # Create separate dataframes for each study
  dataframes_for_each_study = {}
  for study_number in data.Study.unique():
    if not np.isnan(study_number) and study_number not in exluded_study_labels:
      data_for_study = data[data["Study"] == study_number]

      # Drop empty columns, then take complete cases (null values are coded as ' ', need to change to nan)
      # ToDo: Ethnicity should be categorical
      data_for_study.replace(' ', np.nan, regex=True, inplace=True)
      data_for_study.dropna(axis=1, how='all', inplace=True)
      data_for_study.dropna(axis=0, how='any', inplace=True)
      data_for_study = data_for_study.applymap(float)
      data_for_study = data_for_study[data_for_study["newcomb_combined"] != 0.0]
      print("Study {} data size {}".format(study_number, data_for_study.shape))

      if data_for_study.shape[0] > 0:
        X_for_study = data_for_study.drop(labels=["newcomb_combined", "Study"], axis=1)
        X_for_study['payoff1'] = PAYOFF_DICT[study_number][0]
        X_for_study['payoff2'] = PAYOFF_DICT[study_number][1]
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