#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories
SAVE_DIR_ROOT="/preprocessedData/"
SAVE_IMAGE_DIR="$SAVE_DIR_ROOT/preprocessedVal/"
SAVE_DIR_MODEL_FILES="$SAVE_IMAGE_DIR/model_data/"
LOG_DIR="/modelState/"
INPUT_DIR="/inferenceData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState/"

echo "Inferece for Sub Challenge 1"


echo "Making directories for all of the model data etc."
mkdir -p $SAVE_IMAGE_DIR

echo "Creating directory for features found during the preprocessing stage"
mkdir -p $SAVE_DIR_MODEL_FILES

echo "Creating directory for log file"
mkdir -p $LOG_DIR


echo "looking at root directory"
ls /
echo "Looking at modelState"
ls /modelState/

echo "Running Inference Script"
python main_thread.py -v -p -i $INPUT_DIR -s $SAVE_IMAGE_DIR -l $LOG_DIR -m $METADATA_DIR --model $MODEL_STATE_DIR --sub 2 --pca 20 -c 

echo "DONE"
echo "Checking the output directory"
ls /output/

echo "Inference Run Successfully. You may now DANCE :)"
