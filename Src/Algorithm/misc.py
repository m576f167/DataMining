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
