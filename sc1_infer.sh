#!/usr/bin/env bash

#variables that can be used for setting paths and creating directories
SAVE_DIR_ROOT="/preprocessedData"
SAVE_IMAGE_DIR="$SAVE_DIR_ROOT/preprocessedTrain/"
SAVE_DIR_MODEL_FILES="$SAVE_IMAGE_DIR/model_data/"
LOG_DIR="/modelState"
INPUT_DIR="/inferenceData/"
METADATA_DIR="/metadata/"
MODEL_STATE_DIR="/modelState"

echo "Inferece for Sub Challenge 1"
echo "Running Inference Script"
python main_thread.py -t -p -i $INPUT_DIR -s $SAVE_IMAGE_DIR -l $LOG_DIR -m $METADATA_DIR --model $SAVE_DIR_MODEL_FILES --sub 1 --pca 20

echo "DONE"
echo "Inference Run Successfully. You may now DANCE :)"
