#!/bin/bash

set -e

cd data
# make directory to store processed data
mkdir -p  processed
cd ../

# process data from .csv files
python source/ProcessData.py --dataset data/ --processed_dir data/processed --num-files-merge 1 --num-proc 1