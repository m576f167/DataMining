#!/usr/bin/python3
"""
/**
 ******************************************************************************
 * @file dominant_attribute.py
 * @author Mohammad Isyroqi Fathan
 * $Rev: 1 $
 * $Date: Sat Nov 17 15:12:04 CST 2018 $
 * @brief Functions related to Dominant Attribute Algorithm
 ******************************************************************************
 * @copyright
 * @internal
 *
 * @endinternal
 *
 * @details
 * This file contains the functions related to Dominant Attribute Algorithm
 * @ingroup Algorithm
 */
"""

import pandas as pd
import numpy as np
from misc import *
from copy import deepcopy

def generateDiscretizedValue(value, point_ranges, min_value, max_value):
    """
    Generate discrete symbol based on range

    This function computes the discrete symbol representation of the value based
    on the point ranges supplied. The representation will be lowerRange..upperRange.
    Note: point_ranges must be sorted already.

    Parameters
    ----------
    value: float
        The value to be transformed
    point_ranges: list
        List of ranges
    min_value: float
        The minimum value of the range
    max_value: float
        The maximum value of the range

    Returns
    -------
    str
        Discretize symbol representation of value
    """
    # No ranges
    if (len(point_ranges) == 0):
        return ("{:6.3f}..{:6.3f}".format(min_value, max_value))

    # Value is below all point_ranges
    if (value < point_ranges[0]):
        return ("{:6.3f}..{:6.3f}".format(min_value, point_ranges[0]))

    # value is between point_ranges
    for i in range(1, len(point_ranges)):
        if (value < point_ranges[i]):
            return("{:6.3f}..{:6.3f}".format(point_ranges[i - 1], point_ranges[i]))

    # value is above all point_ranges
    return ("{:6.3f}..{:6.3f}".format(point_ranges[len(point_ranges) - 1], max_value))

def generateDiscretizedDataFrame(df, chosen_cut_points):
    """
    Generate discretized data frame based on cut_points

    This function generates discretized data frame based on chosen_cut_points.
    The decision column will always be the last column.

    Parameters
    ----------
    df: pandas.DataFrame
        The data frame to be discretized
    chosen_cut_points: dict
        Dictionary of cut points with the attribute names as the dictionary keys

    Returns
    -------
    pandas.DataFrame
        Discretize data frame
    """
    decision_colname = df.columns[-1]
    attribute_colnames = df.columns[:-1]
    numerical_attribute_colnames = df[attribute_colnames].select_dtypes(include = "number").columns
    non_numerical_attribute_colnames = attribute_colnames.drop(numerical_attribute_colnames)
    df_discretized = pd.DataFrame()
    df_discretized[non_numerical_attribute_colnames] = df[non_numerical_attribute_colnames]
    for attribute in chosen_cut_points.keys():
        current_attribute = df[attribute]
        unique_current_attribute_values = current_attribute.unique()
        df_discretized[attribute] = current_attribute.apply(generateDiscretizedValue,
                                                            point_ranges = chosen_cut_points[attribute],
                                                            min_value = np.min(unique_current_attribute_values),
                                                            max_value = np.max(unique_current_attribute_values))
    df_discretized[decision_colname] = df[decision_colname]
    return (df_discretized)

def computeConditionalEntropy(df, grouping_criteria):
    """
    Compute Conditional Entropy value of the data frame based on grouping_criteria

    This function computes the conditional entropy value of a data frame based on
    grouping_criteria. The value will be conditional entropy of decision (last column)
    conditioned on the group.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset
    grouping_criteria: list
        Grouping criteria of the data frame to be pased to pandas.DataFrame.groupby() function

    Returns
    -------
    float
        Conditional Entropy value
    """
    df_grouped = df.groupby(grouping_criteria)
    num_each_group = df_grouped.apply(lambda x: x.shape[0])
    num_total = num_each_group.sum()

    conditional_entropy_each_group = df_grouped.apply(
        lambda group: (group.groupby(group.columns[-1]).apply(
            lambda x: x.shape[0]
        )/group.shape[0]).apply(lambda x: x * np.log2(x))).groupby(level = 0).apply(
            lambda x: np.sum(x)
        ).T.apply(lambda x: np.sum(x))

#    conditional_entropy_each_group = df_grouped.apply(
#        # Group the groups further by decision
#        lambda group: group.groupby(group.columns[-1]).apply(
#            lambda x: x.shape[0]
#        )/group.shape[0]).groupby(level = 0).apply(
#            lambda x: np.sum(x * np.log2(x))
#        )
    if not (list(conditional_entropy_each_group.index) == list(dict(df_grouped.groups).keys())):
        print("Found")

    conditional_entropy = np.sum(-(num_each_group/num_total) * conditional_entropy_each_group)
    return (conditional_entropy)

def dominantAttribute(df):
    """
    Perform Dominant Attribute Algorithm

    This function performs Dominant Attribute Algorithm on the dataset in data frame
    format

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset

    Returns
    -------
    pandas.DataFrame
        Resulting dataset after performing Dominant Attribute Algorithm
    """
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    df_numerical_attributes = df[attribute_colnames].select_dtypes(include = "number")

    # No numerical attribute
    if (df_numerical_attributes.shape[1] == 0):
        return(df)

    numerical_attribute_colnames = df_numerical_attributes.columns
    df_numerical_attributes = pd.concat([df_numerical_attributes, df[decision_colname]], axis = 1)

    print("------------------------------------------")
    print(" = Dominant Attribute: Numerical attributes are {}".format(list(numerical_attribute_colnames)))

    is_consistent = False
    list_subset = [df_numerical_attributes]
    chosen_cut_points = {}
    total_cut_point = 0

    print("------------------------------------------")
    print(" = Dominant Attribute: Start finding cut points")
    while not is_consistent:
        # Note: Check if it is needed
        if (len(list_subset) == 0):
            break
        current_subset = list_subset.pop(0)
        dominant_attribute = numerical_attribute_colnames[np.argmin([computeConditionalEntropy(current_subset, column)
                                                                     for column in numerical_attribute_colnames]) if len(numerical_attribute_colnames) > 1 else 0]
        print(" = Dominant Attribute: Found dominant attribute = " + dominant_attribute)
        unique_values = list(current_subset.groupby(dominant_attribute).groups)
        # Note: Check if it is needed
        if (len(unique_values) == 1):
            print(" = Dominant Attribute: Only one value, cannot compute possible cut point. Skipping")
            continue
        cut_points = [(unique_values[i] + unique_values[i + 1])/2 for i in range(len(unique_values) - 1)]
        best_cut_point = cut_points[(np.argmin([computeConditionalEntropy(current_subset,
                                                                          np.where(current_subset[dominant_attribute] < cut_point,
                                                                                   True,
                                                                                   False))
                                                for cut_point in cut_points])) if len(cut_points) > 1 else 0]
        print(" = Dominant Attribute: Found best cut point = " + str(best_cut_point))

        # Subset the current dataset further
        new_subsets = dict(current_subset.groupby(np.where(current_subset[dominant_attribute] < best_cut_point, True, False)).__iter__())
        for subset in new_subsets.keys():
            list_subset.append(new_subsets[subset])

        # Append best cut point of dominant attribute
        if (chosen_cut_points.get(dominant_attribute) is not None):
            chosen_cut_points[dominant_attribute] = np.sort(np.append(chosen_cut_points[dominant_attribute], best_cut_point))
        else:
            chosen_cut_points[dominant_attribute] = np.array([best_cut_point])
        total_cut_point += 1

        # Generate tmp_df
        tmp_df = generateDiscretizedDataFrame(df, chosen_cut_points)

        # Check consistency level
        consistency_level = levelOfConsistency(tmp_df)
        is_consistent = consistency_level == 1.0
        print(" = Dominant Attribute: Current consistency level = " + str(consistency_level))

    print("------------------------------------------")
    print(" = Dominant Attribute: Found cut points = {}".format(chosen_cut_points))
    print("------------------------------------------")
    # Merging
    print(" = Dominant Attribute: Start merging")
    for attribute in chosen_cut_points.keys():
        i = 0
        total_element = len(chosen_cut_points[attribute])
        while (i < total_element):
            if (total_cut_point <= 1):
                break
            tmp_chosen_cut_points = deepcopy(chosen_cut_points)
            current_cut_point = tmp_chosen_cut_points[attribute][i]
            tmp_chosen_cut_points[attribute] = np.delete(tmp_chosen_cut_points[attribute], i)
            tmp_df = generateDiscretizedDataFrame(df, tmp_chosen_cut_points)
            consistency_level = levelOfConsistency(tmp_df)
            is_consistent = consistency_level == 1.0
            if (is_consistent):
                # Keep current tmp_chosen_cut_points
                chosen_cut_points = tmp_chosen_cut_points
                total_element = len(chosen_cut_points[attribute])
                print(" = Dominant Attribute: Found redundant cut point = {} {}".format(attribute, str(current_cut_point)) )
                total_cut_point -= 1
            else:
                # Check next element
                i += 1

    # Finalize cut points for all numerical attributes
    for attribute in numerical_attribute_colnames:
        if (chosen_cut_points.get(attribute) is None):
            chosen_cut_points[attribute] = []

    print("------------------------------------------")
    print(" = Dominant Attribute: Finalized cut points = {}".format(chosen_cut_points))
    print("------------------------------------------")
    df_discretized = generateDiscretizedDataFrame(df, chosen_cut_points)
    df[df_discretized.columns] = df_discretized
    return (df)
