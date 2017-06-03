# cd github/camcan_analysis

import os
import os.path as op

import pandas as pd
import mne
from sklearn.preprocessing import Imputer

from camcan.datasets import load_camcan_behavioural
from camcan.datasets import load_camcan_behavioural_feature

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

import numpy as np


data_directory = "/storage/workspace/dengemann/camcan/cc700-scored"

path_data = op.join(data_directory, "total_score.csv")
path_features_map = op.join(data_directory, "behavioural_features.json")
path_participant_info = op.join(data_directory, "participant_data.csv")

scores = [
    "BentlonFaces",
    "CardioMeasures",  # no info
    "Cattell",
    "EkmanEmHex",
    "EmotionalMemory",  # None
    "EmotionRegulation",  # 2 rows only
    "FamousFaces",
    "ForceMatching",
    "Hotel",
    "MotorLearning", # None
    "PicturePriming", # None
    "Proverbs",
    "Synsem",
    "TOT",
    "VSTMcolour",
    # "RTchoice",
    # "RTsimple"
    ]


def get_df_scores(name_experience, path_features_map, path_data,
                  path_participant_info):
    """
    Function to make a dataframe from an experiment ready to be plotted
    :param name_experience: str,
                name of the experience to export
    :param path_features_map: str,
                path of the json file containing the map of the features
    :param path_data: str,
                path of csv file containing all the behavioural datasets
    :param path_participant_info: str,
                path of the csv file containing the variable age
    :return: dataframe
    """
    features_exp = load_camcan_behavioural_feature(
        name_experiment=name_experience,
        exp_feat_map_json=path_features_map)
    features_to_load = ["Observations"] + [c for c in features_exp]
    dataset = load_camcan_behavioural(filename_csv=path_data,
                                      patients_info_csv=path_participant_info,
                                      patients_excluded=None,
                                      column_selected=features_to_load)
    X = dataset.data
    y = dataset.scores.age
    features_to_plot = [feat for feat in features_exp
                        if X[feat].dtypes != object]
                        # and X[feat].isnull().sum() < 100]
    if len(features_to_plot) == 0:
        print('no feratures selected')
        return None
    else:
        print('%i out of %i features selected' %
              (len(features_to_plot), len(features_exp)))
        X = X[features_to_plot]

    # imp = Imputer()
    # X = imp.fit_transform(X)
    X = pd.DataFrame(data=X,
                     columns=features_to_plot)
    X["age"] = y
    return X


def my_bootstrap(x, age, n_iter=2000):
    bootstraps = list()
    for iteration in range(n_iter):
        index = list(range(len(x)))
        bs_index = np.random.choice(
            index, len(x), replace=True)
        xs = lowess(x[bs_index], age[bs_index], return_sorted=False)
        bootstraps.append((xs, x[bs_index], age[bs_index]))
    return np.array(bootstraps)


def get_my_result(df, column, name=None):

    x, age = df[column].values, df['age'].values
    x = MinMaxScaler().fit_transform(x[:, None])[:, 0]
    xs = lowess(x, age, return_sorted=False)
    xs_bs = my_bootstrap(x=x, age=age)
    result = dict(
        name=column if name is None else name,
        xs_bs=xs_bs, x=x, xs=xs, age=age)
    return result


def read_data(score):
    df = get_df_scores(score, path_features_map, path_data,
                       path_participant_info)
    df = df.dropna()
    return df


def check_score(score, selection):
    if score == selection and score not in results:
        print('matched %s' % selection)
        return True
    else:
        return False


results = dict()

for score in scores:
    df = read_data(score)
    if check_score(score, 'BentonFaces'):
        column = 'TotalScore'
        results[score] = (get_my_result(df, column),)
    elif check_score(score, 'CardioMeasures'):
        # Three physiologically distinct markers.
        for column in ('pulse_mean', 'bp_sys_mean', 'bp_dia_mean'):
            results[score] = (get_my_result(df, column),)
    elif check_score(score, 'Cattell'):
        column = 'TotalScore_y'
        results[score] = (get_my_result(df, column),)
    elif check_score(score, 'EkmanEmHex'):
        # average facial expression discrimination and other
        # performance markers
        this_result = list()
        for pattern in ('_Acc', '_MinRT'):
            subcol = pattern[1:]
            df_ = pd.DataFrame(
                {subcol: df.filter(regex=pattern).mean(1),
                 'age': df['age']})
            this_result.append(
                get_my_result(df_, column=subcol)
            )
        results[score] = tuple(this_result)
    elif check_score(score, 'EmotionalMemory'):
        # compute sort of variance over valances
        # differences in neutral and emotional memory
        # are characteristic of age for the 3 different
        # memory types probed.
        this_result = list()
        for pattern in ('PriPr*', 'ValPr*', 'ObjPr*'):
            subcol = pattern[:-1]
            df_ = pd.DataFrame(
                {subcol: df.filter(regex=pattern).mean(1),
                 'age': df['age']})
            this_result.append(
                get_my_result(df_, column=subcol)
            )
        results[score] = tuple(this_result)
    elif check_score(score, 'EmotionRegulation'):
        # compute emotional reactivity by considering the difference
        # in responses to positive and negative films with regard
        # to positive or negative emotions. (should match).
        # Then compute regulation by assessing if people successfully
        # reppraised negative content by comparing negative emtions
        # in negative films naively watched and reappraised.

        this_result = list()
        subcol = 'reactivity_neg_emo'  # bigger is better
        df_ = pd.DataFrame(
            {subcol: df['NegW_mean_neg'] - df['NegW_mean_pos'],
             'age': df['age']})
        this_result.append(
            get_my_result(df_, column=subcol))

        subcol = 'reactivity_pos_emo'  # biggers is better
        df_ = pd.DataFrame(
            {subcol: df['PosW_mean_pos'] - df['PosW_mean_neg'],
             'age': df['age']})
        this_result.append(
            get_my_result(df_, column=subcol))

        subcol = 'regulation_neg_emo'  # biggers is better
        df_ = pd.DataFrame(
            {subcol: df['NegW_mean_neg'] - df['NegR_mean_neg'],
             'age': df['age']})
        this_result.append(
            get_my_result(df_, column=subcol))
        results[score] = tuple(this_result)
    elif check_score(score, 'FamousFaces'):
        # check for capacity to remember details about familiar
        # persons. Account for total number of familiar persons
        # given max of 30 trials.
        individual_max = 30 - df['FacesTest_FAMunfam']
        subcol = 'familiar_faces_details'

        details = np.mean(
            [df['FacesTest_FAMocc'] / individual_max,
             df['FacesTest_FAMnam'] / individual_max], 0)
        df_ = pd.DataFrame(
            {subcol: details, 'age': df['age']})
        results[score] = (get_my_result(df_, column=subcol),)
    elif check_score(score, 'ForceMatching'):
        # get force matching for slider (indirect) and lever (direct)
        # conditions. Both have different distributions
        this_result = list()
        subcol = 'force_match_direct'
        df_ = pd.DataFrame(  # map to log and avoid nan
            {subcol: np.nan_to_num(
                np.log10(df['FingerOverCompensationMean'])),
             'age': df['age']})

        this_result.append(get_my_result(df_, column=subcol))

        subcol = 'force_match_indirect'
        df_ = pd.DataFrame(
            {subcol: df['SliderOverCompensationMean'],
             'age': df['age']})
        this_result.append(get_my_result(df_, column=subcol))

        results[score] = tuple(this_result)
    elif check_score(score, 'Hotel'):
        # get difference from optimal time allocation for 5 uncompletable
        # tasks
        column = 'Time'
        results[score] = (get_my_result(df, column=column),)
    elif check_score(score, 'MotorLearning'):
        # motor learning
        this_result = list()
        subcol = 'trajectory_error'
        df_ = pd.DataFrame(
            {subcol: df.filter(regex='TrajectoryErrorMean*').std(1),
             'age': df['age']})
        this_result.append(get_my_result(df_, column=subcol))
        subcol = 'movement_time'
        df_ = pd.DataFrame(
            {subcol: df.filter(regex='MovementTimeMean*').std(1),
             'age': df['age']})
        this_result.append(get_my_result(df_, column=subcol))
        results[score] = tuple(this_result)
    elif check_score(score, 'PicturePriming'):
        pass  # XXX
    elif check_score(score, 'Proverbs'):
        column = 'Score'
        results[score] = get_my_result(df, column=column)
    elif check_score(score, 'Synsem'):
        this_result = list()
        for pattern in ('ERR*', '_reITRT*'):
            subcol = pattern[:-1]
            df_ = pd.DataFrame(
                {subcol: df.filter(regex=pattern).mean(1),
                 'age': df['age']})
            this_result.append(
                get_my_result(df_, column=subcol)
            )
        results[score] = this_result
    elif check_score(score, 'TOT'):
        column = 'ToT_ratio'
        results[score] = (get_my_result(df, column=column),)
    elif check_score(score, 'VSTMcolour'):
        # unclear how to interpret K1-4.
        # perceptual control block is maybe uninteresting
        pass

mne.externals.h5io.write_hdf5(
    'scores_univariate.h5', results, overwrite=True
)
