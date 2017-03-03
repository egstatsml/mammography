#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories
SAVE_DIR_ROOT="/preprocessedData"
SAVE_DIR="$SAVE_DIR_ROOT"
SAVE_DIR_MODEL_FILES="$SAVE_DIR_ROOT/model_state/"
LOG_DIR="/modelState"
INPUT_DIR="/trainingData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState"

echo "Creating directory for features found during the preprocessing stage"
mkdir -p $SAVE_DIR_MODEL_FILES


echo "Creating directory for log file"
mkdir -p $LOG_DIR

#compile Cython files
echo "Compiling Cython Files"
#chmod u+x ./compile.sh
#./compile.sh

#python main_thread.py -p -t -i /trainingData -s /preprocessedData -l /modelState -m /metadata -k 1 -d 4 -e 0.001 -b -w 0:1,1:20 -v 100

echo "Running Preprocessing script"
python main_thread.py -p -t -i $INPUT_DIR -s $SAVE_DIR -l $LOG_DIR -m $METADATA_DIR -k 1 -d 4 -e 0.001 -b -w 0:1,1:20 -v 100 -a $MODEL_STATE_DIR

echo "DONE"
echo "Preprocessing Run Successfully. You may now DANCE :)"
