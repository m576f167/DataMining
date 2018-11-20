#!/usr/bin/python3

import sys, getopt

def main(argv):
    # Declare necessary variables
    input_file = ''
    output_file = ''
    # Parse Arguments
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print("main.py -i <input_file> -o <output_file>\n")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("main.py -d <sqlite_database> -i <input_html>\n")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--ofile"):
            output_file = arg
    # Make sure input and output files are not empty
    if ((input_file == '') or (output_file == '')):
        print("Error: Input file and output file must not be empty\n")
        sys.exit(-1)

    # Run the program
    parser = LexisNexisParser()
    parser.parse(input_file, db_file)

if __name__ == "__main__":
    main(sys.argv[1:])
