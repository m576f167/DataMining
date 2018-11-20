#!/usr/bin/python3
"""
/**
 ******************************************************************************
 * @file misc.py
 * @author Mohammad Isyroqi Fathan
 * $Rev: 1 $
 * $Date: Sat Nov 17 15:12:04 CST 2018 $
 * @brief Functions related to various Data Mining operations
 ******************************************************************************
 * @copyright
 * @internal
 *
 * @endinternal
 *
 * @details
 * This file contains the functions related to various Data Mining operations needed
 * throughout the project
 * @ingroup Algorithm
 */
"""

import pandas as pd

def levelOfConsistency(df):
    """
    Compute level of consistency

    This function computes the level of consistency of the dataset in data frame
    format. The last column is considered as the decision, as specified in the
    requirement.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset

    Returns
    -------
    float
        level of consistency score
    """
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    num_data = df.shape[0]

    df_attribute_duplicate = df[df.duplicated(attribute_colnames, keep = False)]
    num_inconsistent = df_attribute_duplicate.groupby(attribute_colnames.tolist())[decision_colname].apply(lambda x: x.shape[0] if x.unique().shape[0] > 1 else 0).sum()

    return ((num_data - num_inconsistent)/num_data)

def upperApproximation(df, concept):
    """
    Generate upper approximation dataset from df

    This function generates the upper approximation of the dataset based on the
    concept. Note: This will modify the original data frame. To get the original
    data frame decision back, assign the original_decision returned to the decision
    column

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset
    concept: String or Numerical
        Concept value for lower approximation. It can be String or Numerical, based
        on the decision data type

    Returns
    -------
    pandas.DataFrame
        Upper approximation of the dataset based on the concept
    pandas.Series
        Pandas Series of original decision
    """
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    original_decision = df[decision_colname].copy()
    # Find cases for which decisions to be replaced with special
    subset_lower_approximation_na = df.groupby(list(attribute_colnames)).filter(lambda x:  (x[decision_colname] == concept).any(), dropna = False)
    subset_lower_approximation_na = subset_lower_approximation_na.apply(lambda x: x.isna().any(), axis = 1)
    # Replace uncertain cases and cases not matching to concet with "SPECIAL_DECISION"
    df.loc[subset_lower_approximation_na, decision_colname] = "SPECIAL_DECISION"
    df.loc[~subset_lower_approximation_na, decision_colname] = concept
    return (df, original_decision)

def lowerApproximation(df, concept):
    """
    Generate lower approximation dataset from df

    This function generates the lower approximation of the dataset based on the
    concept. Note: This will modify the original data frame. To get the original
    data frame decision back, assign the original_decision returned to the decision
    column

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset
    concept: String or Numerical
        Concept value for lower approximation. It can be String or Numerical, based
        on the decision data type

    Returns
    -------
    pandas.DataFrame
        Lower approximation of the dataset based on the concept
    pandas.Series
        Pandas Series of original decision
    """
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    original_decision = df[decision_colname].copy()
    # Find cases for which decisions to be replaced with special
    subset_lower_approximation_na = df.groupby(list(attribute_colnames)).filter(lambda x:  x[decision_colname].iloc[0] == concept if (x[decision_colname].unique().shape[0] == 1) else False, dropna = False)
    subset_lower_approximation_na = subset_lower_approximation_na.apply(lambda x: x.isna().any(), axis = 1)
    # Replace uncertain cases and cases not matching to concet with "SPECIAL_DECISION"
    df.loc[subset_lower_approximation_na, decision_colname] = "SPECIAL_DECISION"
    return (df, original_decision)
