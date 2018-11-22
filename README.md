# DataMining
## Name: Mohammad Isyroqi Fathan
## KUID: 2763993

Data Mining Project using LEM2, Dominant Attribute Discretization, and Rough Set Estimation

## How to Run
To run the program:
	python3 main.py -i <input_fle> -o <output_file> [-r -h]
	
	-i or --ifile: The input file path and name
	-o or --ofile: The output file path and name
	-r or --run: Whether to keep program running to ask for next input and specify next output (default is False)
	-h or --help: Display help message

## Project Structure
The project directory structure is as follows:

.
├── Data
│   └── Test
│       ├── austr.txt
│       ├── bowl.txt
│       ├── breast.txt
│       ├── german.txt
│       ├── hw2.txt
│       ├── hw3.txt
│       ├── iris.txt
│       ├── m-global.txt
│       ├── proj-1.txt
│       └── wine.txt
├── LICENSE
├── README.md
├── Result
│   └── Test
└── Src
    ├── Algorithm
    │   ├── LEM2.py
    │   ├── __init__.py
    │   ├── dominant_attribute.py
    │   └── misc.py
    ├── Util
    │   ├── __init__.py
    │   └── parser.py
    ├── main.py
    ├── test_script.sh
    └── test_script_parallel.sh

Explanation:
	- Data/Test: Contains test datasets
	- Result/Test: Where to store results
	- Src: Contains Source Code
	- LEM2.py: Functions related to LEM2 Algorithm
	- dominant_attribute.py: Functions related to Dominant Attribute Algorithm
	- misc.py: Functions related to project such as lower and upper approximations and level of consistency computation
	- parser.py: Functions related to parsing input file and generating output files
	- main.py: Main program
	- test_script.sh: Example test script to run all the test datasets
	- test_script_parallel.sh: Example test script to run all datasets in parallel using GNU Parallel

## Output Files
The program generates several output files as follows:
	- file.r: Resulting rule set. It can be normal rule set (without comment line) or certain and possible rule sets (with comment line) if the dataset is not consistent
	- file.disc: The discretized dataset if the dataset contains one or more numerical attribute columns
	- file_certain_concept.disc: The discretized dataset corresponding to certain rule set with the corresponding concept
	- file_possible_concept.disc: The discretized dataset corresponding to possible rule set with the corresponding concept

Note that I generate multiple discretized files in the case of inconsistent dataset. This is because when we generate lower approximation or upper approximation dataset corresponding
to a concept, we replace the other values other than current concept with "SPECIAL_DECISION". This results in different values of decision column in different datasets (lower approximation
or upper approximation corresponding to a concept). This may affect the level of consistency calculation when performing the dominant attribute algorithm. Thus, I always run dominant attribute
algorithm per different dataset (lower and upper approximation corresponding to each concept) and output the resulting discretized datasets. Also note that I use the original decision values
when writing these discretized datasets instead of replacing them with certain decision values or possible decision values
