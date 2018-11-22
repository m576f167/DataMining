#!/usr/bin/python3

import sys, getopt
import numpy as np
from Algorithm.misc import *
from Algorithm.dominant_attribute import *
from Algorithm.LEM2 import *
from Util.parser import *

def runPipeline(df, concept, output_file):
    """
    Run processing pipeline

    This function runs the processing pipeline from discretization to output.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame representation of the dataset
    concept: String or Numerical
        Concept value. It can be String or Numerical, based on the decision data type
    output_file: String
        Output file name and path

    Returns
    -------
    list
        Rules generated
    np.array
        Indices of elements in decision concept
    pandas.DataFrame
        Data frame representation of the dataset

    """
    attribute_colnames = df.columns[:-1]
    if (df[attribute_colnames].select_dtypes(include = "number").shape[1] > 0):
        print("|========================================|")
        print(" [#] Running Dominant Attribute Algorithm")
        df = dominantAttribute(df)
    print("|========================================|")
    print(" [#] Running LEM2 Algorithm")
    rules, decision_concept = LEM2(df, concept)
    print("|========================================|")
    print(" [#] Running Dropping Condition Algorithm")
    rules = [droppingCondition(rule, decision_concept) for rule in rules]
    return (rules, decision_concept, df)

def run(input_file, output_file):
    """
    Run the program

    This function runs the main execution part of the program. The program flows
    as follow:
        parse input -> Check consistency -----------------------------------> discretize -> LEM2 -> dropping condition -> output
                                         |-> generate lower and upper sets -^

    Parameters
    ----------
    input_file: String
        Path to input file
    output_file: String
        Path to output files

    Returns
    -------
    """
    df = parseInput(input_file)
    decision_colname = df.columns[-1]
    consistency_level = levelOfConsistency(df)
    concepts = df[decision_colname].unique()
    rule_set_normal = {}
    rule_set_certain = {}
    rule_set_possible = {}
    rule_info_normal = {}
    rule_info_certain = {}
    rule_info_possible = {}
    is_consistent = consistency_level == 1.0
    attribute_colnames = df.columns[:-1]
    is_discretized = True if (df[attribute_colnames].select_dtypes(include = "number").shape[1] > 0) else False
    for concept in concepts:
        if is_consistent:
            # Data set is consistent
            print("|++++++++++++++++++++++++++++++++++++++++|")
            print(" [@]=> Normal Rule Set for ({}, {})".format(decision_colname, concept))
            print("|++++++++++++++++++++++++++++++++++++++++|")
            rules, decision_concept, df = runPipeline(df, concept, output_file)
            print("|========================================|")
            print(" [#] Computing Rule Info")
            rules_info = [computeRuleInfo(rule, decision_concept) for rule in rules]
            rule_set_normal["({}, {})".format(decision_colname, concept)] = rules
            rule_info_normal["({}, {})".format(decision_colname, concept)] = rules_info
        else:
            # Certain Rule Set
            print("|++++++++++++++++++++++++++++++++++++++++|")
            print(" [@]=> Certain Rule Set for ({}, {})".format(decision_colname, concept))
            print("|++++++++++++++++++++++++++++++++++++++++|")
            df_certain, original_decision = lowerApproximation(df.copy(), concept)
            rules, decision_concept, df_certain = runPipeline(df_certain, concept, output_file)
            original_decision_concept = np.where(original_decision == concept)[0]
            print("|========================================|")
            print(" [#] Computing Rule Info")
            rules_info = [computeRuleInfo(rule, original_decision_concept) for rule in rules]
            rule_set_certain["({}, {})".format(decision_colname, concept)] = rules
            rule_info_certain["({}, {})".format(decision_colname, concept)] = rules_info
            df_certain[decision_colname] = original_decision

            # Possible Rule Set
            print("|++++++++++++++++++++++++++++++++++++++++|")
            print(" [@]=> Possible Rule Set for ({}, {})".format(decision_colname, concept))
            print("|++++++++++++++++++++++++++++++++++++++++|")
            df_possible, original_decision = upperApproximation(df.copy(), concept)
            rules, decision_concept, df_possible = runPipeline(df_possible, concept, output_file)
            original_decision_concept = np.where(original_decision == concept)[0]
            print("|========================================|")
            print(" [#] Computing Rule Info")
            rules_info = [computeRuleInfo(rule, original_decision_concept) for rule in rules]
            rule_set_possible["({}, {})".format(decision_colname, concept)] = rules
            rule_info_possible["({}, {})".format(decision_colname, concept)] = rules_info
            df_possible[decision_colname] = original_decision
            if (is_discretized):
                print("|========================================|")
                print(" [#] Outputting Certain Rule Set")
                # Print to file
                file_handler_output = open(output_file + "_certain_{}.disc".format(concept), "w")
                file_handler_output.write("[ " + " ".join("{}".format(x) for x in df_certain.columns) + " ]\n")
                file_handler_output.close()
                df_certain.to_csv(output_file + "_certain_{}.disc".format(concept), sep = " ", index = False, header = False, mode = "a")
                print("|========================================|")
                print(" [#] Outputting Possible Rule Set")
                # Print to file
                file_handler_output = open(output_file + "_possible_{}.disc".format(concept), "w")
                file_handler_output.write("[ " + " ".join("{}".format(x) for x in df_possible.columns) + " ]\n")
                file_handler_output.close()
                df_possible.to_csv(output_file + "_possible_{}.disc".format(concept), sep = " ", index = False, header = False, mode = "a")
    print("|[][][][][][][][][][][][][][][][][][][][]|")
    print("|[][][][][][][][][][][][][][][][][][][][]|")
    if ((is_discretized) and (is_consistent)):
        print("|========================================|")
        print(" [#] Outputting Normal Rule Set")
        # Print to file
        file_handler_output = open(output_file + ".disc", "w")
        file_handler_output.write("[ " + " ".join("{}".format(x) for x in df.columns) + " ]\n")
        file_handler_output.close()
        df.to_csv(output_file + ".disc", sep = " ", index = False, header = False, mode = "a")
    if is_consistent:
        # Write rule set normally
        print("|========================================|")
        print(" [#] Outputting Normal Rule Set")
        outputRuleSet(output_file, rule_set_normal, rule_info_normal)
    else:
        # Write certain rule set
        print("|========================================|")
        print(" [#] Outputting Certain Rule Set")
        file_handler_output = open(output_file + ".r", "w")
        file_handler_output.write("! Certain rule set\n")
        file_handler_output.close()
        outputRuleSet(output_file, rule_set_certain, rule_info_certain, is_rewrite = False)
        # Write possible rule set
        print("|========================================|")
        print(" [#] Outputting Possible Rule Set")
        file_handler_output = open(output_file + ".r", "a")
        file_handler_output.write("\n\n! Possible rule set\n")
        file_handler_output.close()
        outputRuleSet(output_file, rule_set_possible, rule_info_possible, is_rewrite = False)
    return

def promptInput():
    """
    Prompts input from user

    This function prompts input for input file, output file, and whether to run
    again from user

    Parameters
    ----------

    Returns
    -------
    Str
        Path to input file
    Str
        Path to output files
    Bool
        Whether to run the program again
    """
    answer = ""

    # Prompt whether to run again
    while answer not in ("y", "n"):
        try:
            answer = raw_input(" ? Run again (y/n): ")
        except Exception as e:
            print(e)
    is_running = answer == "y"

    # Set default values in case something go wrong
    input_file = ""
    output_file = ""

    # Prompt for input file and output file paths
    try:
        input_file = raw_input(" ? Input file path: ")
        output_file = raw_input(" ? Output file path: ")
    except Exception as e:
        print(e)
    return(input_file, output_file, is_running)


def main(argv):
    # Declare necessary variables
    input_file = ''
    output_file = ''
    is_keep_running = False

    # Parse Arguments
    try:
        opts, args = getopt.getopt(argv, "hi:o:r", ["help", "ifile=", "ofile=", "run"])
    except getopt.GetoptError:
        print("main.py -i <input_file> -o <output_file> -r <run>\n")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("main.py -i <input_file> -o <output_file> [-r]\n")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
        elif opt in ("-r", "--run"):
            is_keep_running = True
    # Make sure input and output files are not empty
    if ((input_file == '') or (output_file == '')):
        print("Error: Input file and output file must not be empty\n")
        sys.exit(-1)

    is_running = True
    # Keep running the program if option is selected
    while is_running:
        try:
            run(input_file, output_file)
        except Exception as e:
            print(e)
        if not is_keep_running:
            break
        input_file, output_file, is_running = promptInput()

if __name__ == "__main__":
    main(sys.argv[1:])
