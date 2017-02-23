#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories
SAVE_DIR="preprocessedData/images/"
LOG_DIR="/modelState"
INPUT_DIR="/trainingData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState"

echo "Creating directory for preprocessed files"
mkdir -p $SAVE_DIR

echo "Creating directory for log file"
mkdir -p $LOG_DIR

#compile Cython files
echo "Compiling Cython Files"
#chmod u+x ./compile.sh
./compile.sh

#python main_thread.py -p -t -i /trainingData -s /preprocessedData -l /modelState -m /metadata -k 1 -d 4 -e 0.001 -b -w 0:1,1:20 -v 100

echo "Running Preprocessing script"
python main_thread.py -p -i $INPUT_DIR -s $SAVE_DIR -l $LOG_DIR -m $METADATA_DIR -k 1 -d 4 -e 0.001 -b -w 0:1,1:20 -v 100


echo "DONE"
echo "Preprocessing Run Successfully. You may now DANCE :)"
