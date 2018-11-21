#!/usr/bin/python3
"""
/**
 ******************************************************************************
 * @file parser.py
 * @author Mohammad Isyroqi Fathan
 * $Rev: 1 $
 * $Date: Sat Nov 17 15:12:04 CST 2018 $
 * @brief Functions related to parsing the input file and producing output file
 ******************************************************************************
 * @copyright
 * @internal
 *
 * @endinternal
 *
 * @details
 * This file contains the functions for parsing the input file and producing output
 * file. It is based on LERS format
 *
 * @ingroup util
 */
"""

import pandas as pd
from enum import Enum

class ParseState(Enum):
    """
    States for parseInput
    """
    START = 0
    VARIABLE = 1
    ATTRIBUTE = 2
    DATA = 3
    COMMENT = 4
    END = 5

def parseInput(input_file):
    """
    Parse LERS format input file

    This function parse input file in LERS format and generate pandas dataframe of
    the dataset

    Parameters
    ----------
    input_file: string
        Input file name and path

    Returns
    -------
    pandas.DataFrame
        Pandas dataframe of the parsed input file
    """
    file_handler_input = open(input_file, "r")
    read_lines = file_handler_input.readlines()
    file_handler_input.close()
    current_state = ParseState.START
    prev_state = current_state
    current_data = []

    variables = None
    attributes = None
    df = None
    current_attribute = 0
    num_attributes = 0

    for line in read_lines:
        if (current_state == ParseState.COMMENT):
            current_state = prev_state
        for token in line.split():
            if (current_state == ParseState.START):
                current_data = []
                if (token == "<"):
                    current_state = ParseState.VARIABLE
                elif (token == "["):
                    current_state = ParseState.ATTRIBUTE
                elif (token == "!"):
                    prev_state = current_state
                    current_state = ParseState.COMMENT
                else:
                    if (attributes is None):
                        raise(Exception(" ! Syntax Error: Attributes have not been read"))
                    current_attribute = 0
                    current_state = ParseState.DATA
            elif (current_state == ParseState.VARIABLE):
                if (token == ">"):
                    variables = current_data
                    current_state = ParseState.START
                elif ((token == "a") or
                      (token == "x") or
                      (token == "d")):
                    current_data.append(token)
                else:
                    raise(Exception(" ! Syntax Error: Unrecognized token for Variables"))
            elif (current_state == ParseState.ATTRIBUTE):
                if (token == "]"):
                    attributes = current_data
                    df = pd.DataFrame([], columns = attributes)
                    num_attributes = len(current_data)
                    current_state = ParseState.START
                else:
                    current_data.append(token)
            elif (current_state == ParseState.COMMENT):
                pass
            elif (current_state == ParseState.END):
                pass
            # Handle Data
            if (current_state == ParseState.DATA):
                try:
                    token = float(token)
                except:
                    pass
                current_data.append(token)
                current_attribute += 1
                if (current_attribute == num_attributes):
                    df = df.append(pd.DataFrame([current_data], columns = attributes))
                    current_state = ParseState.START
    # Check for missing data
    if ((current_attribute > 0) and (current_attribute < num_attributes)):
        raise(Exception(" ! Missing Data: It is possible that the dataset is missing {} data attributes".format(num_attributes - current_attribute)))

    df = df.reset_index(drop = True)
    return (df)

def outputRuleSet(output_file, rule_set, rule_info, is_rewrite = True):
    """
    Write rule set to output file

    This function writes the rule set to specified output file. The rule_set has
    the format specified in parameter section.

    Parameters
    ----------
    output_file: String
        Output file name and path
    rule_set: dict
        A dictionary representing the rule set with the following format:
            {(decision_column_name, decision_value) : rules}
    rule_info: dict
        A dictionary representing the rule info for the rule set with the following
        format:
            {(decision_column_name, decision_value) : [(x, y, z), (x, y, z), ...]}
        with the x = specificity, y = strength, z = total number of matching cases
    is_rewrite: bool
        Whether to rewrite the file or not

    Returns
    -------
    """
    file_handler_output = open(output_file + ".r", "w") if is_rewrite else open(output_file + ".r", "a")
    for concept, rules in rule_set.items():
        infos = rule_info[concept]
        for rule_index, rule in enumerate(rules):
            info = infos[rule_index]
            rule_info_string = str(info[0]) + ", " + str(info[1]) + ", " + str(info[2]) + "\n"
            conditions = list(rule.keys())
            rule_string = str(conditions[0])
            for i in range(1, len(conditions)):
                rule_string = rule_string + " & " + str(conditions[i])
            rule_string = rule_string + " -> " + str(concept) + "\n"
            file_handler_output.write(rule_info_string)
            file_handler_output.write(rule_string)
    file_handler_output.close()
    return
