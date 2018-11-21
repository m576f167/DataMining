#!/bin/bash

ls ../Data/Test | parallel "python3 main.py -i ../Data/Test/{} -o ../Result/Test/{.}"
