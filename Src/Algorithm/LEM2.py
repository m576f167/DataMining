#!/usr/bin/python3
"""
/**
 ******************************************************************************
 * @file LEM2.py
 * @author Mohammad Isyroqi Fathan
 * $Rev: 1 $
 * $Date: Sat Nov 17 15:12:04 CST 2018 $
 * @brief Functions related to LEM2 algorithm
 ******************************************************************************
 * @copyright
 * @internal
 *
 * @endinternal
 *
 * @details
 * This file contains the functions related to LEM2 Algorithm
 * @ingroup Algorithm
 */
"""

import pandas as pd
import numpy as np
from functools import reduce

def LEM2(df, concept):
    """
    Perform LEM2 Algorithm

    This function performs LEM2 Algorithm on the dataset in data frame format. The
    concept can be String or Numerical depending on decision column data type

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset
    concept: String or Numerical
        Concept value. It can be String or Numerical, based on the decision data type

    Returns
    -------
    list
        List of dictionary containing rules for local covering of the concept
    np.array
        Indices of elements in decision concept
    """
    attribute_colnames = df.columns[:-1]
    decision_colname = df.columns[-1]
    attribute_value_pairs = {}
    decision_concept = np.where(df[decision_colname] == concept)[0]

    print("------------------------------------------")
    print(" = LEM2: Generating attribut-value pairs")
    # Generate attribute-value pairs
    for attribute in attribute_colnames:
        df_grouped = df.groupby(attribute).indices
        for value in df_grouped.keys():
            attribute_value_pairs["({}, {})".format(attribute, value)] = df_grouped[value]

    goal = decision_concept
    local_covering = []
    num_goal = len(goal)
    print("------------------------------------------")
    print(" = LEM2: Finding Local Covering")
    while (num_goal > 0):
        selected_pairs = {}
        possible_pairs = {k:v for k, v in attribute_value_pairs.items() if np.isin(v, goal).any()}

        num_selected_pairs = len(selected_pairs)
        is_selected_pairs_subset_concept = False
        while ((num_selected_pairs == 0) or (not is_selected_pairs_subset_concept)):
            # Choose maximum intersection
            max_size_intersection = np.max([len(value) for value in possible_pairs.values()])
            pair_selected = {k:v for k, v in possible_pairs.items() if len(v) == max_size_intersection}
            if (len(pair_selected) > 1):
                # Choose minimum min_cardinality
                min_cardinality = np.min([len(value) for value in pair_selected.values()])
                pair_selected = {k:v for k, v in pair_selected.items() if len(v) == min_cardinality}
                if (len(pair_selected) > 1):
                    # Choose the first key heuristically
                    pair_key = list(pair_selected.keys())[0]
                    pair_selected = {pair_key: pair_selected[pair_key]}
            pair_key, pair_value = list(pair_selected.items())[0]
            selected_pairs[pair_key] = pair_value
            goal = np.intersect1d(pair_value, goal)
            possible_pairs = {k:v for k, v in attribute_value_pairs.items() if np.isin(v, goal).any()}
            possible_pairs = {k:v for k, v in possible_pairs.items() if selected_pairs.get(k) is None}
            num_selected_pairs = len(selected_pairs)
            is_selected_pairs_subset_concept = np.isin(reduce(np.intersect1d, selected_pairs.values()), decision_concept).all()
        list_pair_key = list(selected_pairs.keys())
        for pair_key in list_pair_key:
            pair_value = selected_pairs[pair_key]
            del selected_pairs[pair_key]
            num_pair = len(selected_pairs)
            if ((num_pair == 0) or (not np.isin(reduce(np.intersect1d, selected_pairs.values()), decision_concept).all())):
                selected_pairs[pair_key] = pair_value
        local_covering.append(selected_pairs)
        union_local_covering = reduce(np.union1d, [reduce(np.intersect1d, selected_pairs.values()) for selected_pairs in local_covering])
        goal = [item for item in decision_concept if not np.isin(item, union_local_covering)]
        num_goal = len(goal)
    i = 0
    total_element = len(local_covering)
    print("------------------------------------------")
    print(" = LEM2: Finding Maximum Local Covering")
    while (i < total_element):
        selected_pairs = local_covering.pop(i)
        union_local_covering = reduce(np.union1d, [reduce(np.intersect1d, selected_pairs.values()) for selected_pairs in local_covering])
        if np.array_equal(union_local_covering, decision_concept):
            total_element = len(local_covering)
        else:
            local_covering.insert(i, selected_pairs)
            i += 1
    return (local_covering, decision_concept)
