# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = CTG_features.drop(columns=[extra_feature]).apply(lambda x: pd.to_numeric(x, errors='coerce'))
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.drop(columns=[extra_feature]).apply(lambda x: pd.to_numeric(x, errors='coerce'))
    for col in list(c_cdf.columns):
        try:
            a = np.array(c_cdf[col].value_counts().index)
            size = c_cdf[col].value_counts(dropna=False).loc[np.nan]
            p = np.array(c_cdf[col].value_counts(normalize=True))

            sample = np.random.choice(a, size=size, p=p)
            index_nan = list(c_cdf[c_cdf[col].isna()].index)
            for i in range(len(index_nan)):
                c_cdf.loc[index_nan[i], col] = sample[i]
        except Exception:
            pass
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    for col in c_feat.columns:
        d_summary[col] = dict(c_feat[col].describe().loc[['min', '25%', '50%', '75%', 'max']])
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_no_outlier = c_feat.copy()
    for col in c_no_outlier.columns:
        IQR = d_summary[col]['75%']-d_summary[col]['25%']
        c_no_outlier[col]=c_no_outlier[col].where(c_no_outlier[col].between((d_summary[col]['25%']-1.5*IQR),(d_summary[col]['75%']+1.5*IQR)))
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[feature].where(c_cdf[feature] <= thresh)
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    if mode == 'standard':
        nsd_res = ((CTG_features - CTG_features.mean()) / CTG_features.std())

    elif mode == 'MinMax':
        nsd_res = ((CTG_features - CTG_features.min()) / (CTG_features.max() - CTG_features.min()))

    elif mode == 'mean':
        nsd_res = ((CTG_features - CTG_features.mean()) / (CTG_features.max() - CTG_features.min()))
    else:
        nsd_res = CTG_features

    if flag == True:
        nsd_res[x].plot.hist(bins=100, figsize=(10, 5))
        plt.title(x + mode)
        plt.xlabel('beats/min')
        plt.ylabel('Count')
        plt.show()

        nsd_res[y].plot.hist(bins=100, figsize=(10, 5))
        plt.title(y + mode)
        plt.xlabel('%')
        plt.ylabel('Count')
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
